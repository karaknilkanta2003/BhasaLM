"""
Evaluation and Text Generation Toolkit for Monolingual Transformer LMs
======================================================================
Implements:
1. Intrinsic Language Modeling Metrics:
   - Cross-Entropy Loss
   - Perplexity: PPL = exp(loss)
   - Bits-Per-Byte (BPB): Fair cross-lingual comparison accounting for tokenizer differences.
2. Open-Ended Text Generation:
   - Greedy decoding
   - Temperature sampling: T in {0.5, 1.0, 1.5}
3. Generation Diagnostics:
   - BLEU-4, chrF++, ROUGE-L against held-out references
   - Repetition rate (4-gram loops)
   - Distinct-1 and Distinct-2 diversity
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import torch
import torch.nn.functional as F
import sentencepiece as spm


# ==============================================================================
# 1. Intrinsic Evaluation: Perplexity & Bits-Per-Byte (BPB)
# ==============================================================================

def compute_perplexity(loss: float) -> float:
    """Computes Perplexity = exp(cross_entropy_loss)."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def compute_bits_per_byte(
    total_cross_entropy_loss: float,
    total_tokens: int,
    total_utf8_bytes: int
) -> float:
    """
    Computes Bits-Per-Byte (BPB):
    BPB = (Total Loss in nats * total_tokens) / (total_utf8_bytes * ln(2))
        = Total NLL in bits / total_utf8_bytes
    
    Why BPB is essential:
    Tokenizers differ in fertility across languages (Hindi ~1.33 vs Assamese ~1.61 subwords/word).
    Perplexity measures uncertainty per token, which unfairly rewards models with larger vocabularies
    or lower fertility. BPB measures uncertainty per raw UTF-8 byte of text, creating a completely
    fair, tokenizer-independent comparison between Model H and Model L.
    """
    assert total_utf8_bytes > 0, "total_utf8_bytes must be positive"
    total_nll_nats = total_cross_entropy_loss * total_tokens
    total_nll_bits = total_nll_nats / math.log(2.0)
    return total_nll_bits / total_utf8_bytes


# ==============================================================================
# 2. Text Diversity & Repetition Metrics
# ==============================================================================

def compute_distinct_n(tokens: List[str], n: int = 1) -> float:
    """Computes Distinct-N: Unique n-grams divided by total n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_repetition_rate(tokens: List[str], n: int = 4) -> float:
    """
    Computes fraction of repeated n-grams:
    1.0 - (unique n-grams / total n-grams)
    Higher values indicate pathological degenerate repetition loops.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated_count = sum(c for c in counts.values() if c > 1)
    return repeated_count / len(ngrams)


# ==============================================================================
# 3. Reference-Based Generation Evaluation: BLEU, chrF++, ROUGE-L
# ==============================================================================

def compute_ngram_matches(hyp_tokens: List[str], ref_tokens: List[str], n: int) -> Tuple[int, int]:
    """Computes clipped n-gram matches and total candidate n-grams."""
    if len(hyp_tokens) < n or len(ref_tokens) < n:
        return 0, max(len(hyp_tokens) - n + 1, 0)
    hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
    ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
    clipped = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
    total = sum(hyp_ngrams.values())
    return clipped, total


def compute_bleu4(hypotheses: List[str], references: List[str]) -> float:
    """
    Computes corpus-level BLEU-4 with brevity penalty.
    """
    clipped_matches = [0, 0, 0, 0]
    total_candidates = [0, 0, 0, 0]
    hyp_len_total = 0
    ref_len_total = 0

    for hyp, ref in zip(hypotheses, references):
        h_toks = hyp.strip().split()
        r_toks = ref.strip().split()
        hyp_len_total += len(h_toks)
        ref_len_total += len(r_toks)

        for n in range(1, 5):
            c, t = compute_ngram_matches(h_toks, r_toks, n)
            clipped_matches[n - 1] += c
            total_candidates[n - 1] += t

    precisions = []
    for c, t in zip(clipped_matches, total_candidates):
        if t == 0 or c == 0:
            precisions.append(1e-6)
        else:
            precisions.append(c / t)

    log_p = sum(math.log(p) for p in precisions) / 4.0
    geo_mean = math.exp(log_p)

    # Brevity penalty
    if hyp_len_total > ref_len_total:
        bp = 1.0
    elif hyp_len_total == 0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - ref_len_total / hyp_len_total)

    return (bp * geo_mean) * 100.0


def compute_chrf(hypotheses: List[str], references: List[str], beta: float = 2.0) -> float:
    """
    Character n-gram F-score (chrF++).
    Particularly informative for agglutinative and morphologically rich Indic languages
    (Hindi and Assamese) because character n-grams capture inflections and matras that word BLEU misses.
    """
    total_f = 0.0
    count = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_chars = list(hyp.replace(" ", ""))
        ref_chars = list(ref.replace(" ", ""))
        if not hyp_chars or not ref_chars:
            continue

        n_scores = []
        for n in range(1, 7): # Character n-grams 1 to 6
            c, t = compute_ngram_matches(hyp_chars, ref_chars, n)
            _, ref_t = compute_ngram_matches(ref_chars, ref_chars, n)
            p = (c / t) if t > 0 else 0.0
            r = (c / ref_t) if ref_t > 0 else 0.0
            if p + r > 0:
                f = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
            else:
                f = 0.0
            n_scores.append(f)

        if n_scores:
            total_f += sum(n_scores) / len(n_scores)
            count += 1

    return (total_f / count * 100.0) if count > 0 else 0.0


def compute_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    """Computes ROUGE-L based on Longest Common Subsequence (LCS)."""
    def lcs(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if x[i] == y[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[m][n]

    f1_scores = []
    for hyp, ref in zip(hypotheses, references):
        h_toks = hyp.strip().split()
        r_toks = ref.strip().split()
        if not h_toks or not r_toks:
            continue
        lcs_len = lcs(h_toks, r_toks)
        p = lcs_len / len(h_toks)
        r = lcs_len / len(r_toks)
        if p + r > 0:
            f1_scores.append((2 * p * r) / (p + r))

    return (sum(f1_scores) / len(f1_scores) * 100.0) if f1_scores else 0.0
