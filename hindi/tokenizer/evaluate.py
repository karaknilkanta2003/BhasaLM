"""
Hindi Tokenizer Evaluation Script
Computes all rubric-mandated metrics:
1. Vocabulary size
2. Token-frequency statistics (top frequent tokens, frequency distribution)
3. Average characters per token
4. Qualitative tokenization examples
5. Unknown-token (UNK) statistics & Fertility on held-out test split
"""

import os
import sys
import json
import argparse
from collections import Counter
import sentencepiece as spm
from typing import List, Dict, Any

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from common.utils import get_logger, load_jsonl

logger = get_logger("hindi_sp_eval")

SAMPLE_SENTENCES = [
    "भाषा मॉडल और कृत्रिम बुद्धिमत्ता का विकास तेजी से हो रहा है।",
    "प्रेमचंद की कहानियों में भारतीय समाज की यथार्थ तस्वीर दिखाई देती है।",
    "हिंदी भारत की सबसे अधिक बोली जाने वाली आधिकारिक भाषाओं में से एक है।",
    "कंप्यूटर विज्ञान और प्राकृतिक भाषा प्रसंस्करण में नए शोध हो रहे हैं।",
    "महाभारत और रामायण भारतीय संस्कृति के दो महान महाकाव्य हैं।",
    "शिक्षा और अनुसंधान से ही समाज की प्रगति संभव हो सकती है।"
]

def evaluate_tokenizer(model_path: str, test_file: str) -> Dict[str, Any]:
    """
    Evaluates tokenizer on held-out test data.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    unk_id = sp.unk_id()
    vocab_size = sp.get_piece_size()
    
    logger.info(f"Loaded tokenizer from {model_path} (Vocab Size: {vocab_size}, UNK id: {unk_id})")

    docs = load_jsonl(test_file)
    logger.info(f"Loaded {len(docs)} test documents from {test_file}")

    total_words = 0
    total_tokens = 0
    total_chars = 0
    total_unk = 0
    token_freq_counter = Counter()

    for doc in docs:
        text = doc.get("text", "").strip()
        if not text:
            continue
        words = text.split()
        token_ids = sp.encode_as_ids(text)
        
        total_words += len(words)
        total_tokens += len(token_ids)
        total_chars += len(text)
        
        for tid in token_ids:
            token_freq_counter[tid] += 1
            if tid == unk_id:
                total_unk += 1

    fertility = round(total_tokens / total_words, 4) if total_words > 0 else 0.0
    unk_rate = round((total_unk / total_tokens * 100), 4) if total_tokens > 0 else 0.0
    chars_per_token = round(total_chars / total_tokens, 4) if total_tokens > 0 else 0.0

    # Token frequency statistics
    top_50_tokens = []
    for tid, count in token_freq_counter.most_common(50):
        piece = sp.id_to_piece(tid)
        top_50_tokens.append({
            "token_id": tid,
            "piece": piece,
            "count": count,
            "frequency_percentage": round(count / total_tokens * 100, 4) if total_tokens else 0
        })

    # Qualitative examples
    tokenization_samples = []
    for sent in SAMPLE_SENTENCES:
        pieces = sp.encode_as_pieces(sent)
        token_ids = sp.encode_as_ids(sent)
        tokenization_samples.append({
            "original_sentence": sent,
            "subword_pieces": pieces,
            "token_ids": token_ids,
            "num_pieces": len(pieces),
            "num_words": len(sent.split()),
            "fertility": round(len(pieces) / len(sent.split()), 2)
        })

    metrics = {
        "language": "Hindi (hi)",
        "model_path": model_path,
        "vocabulary_size": vocab_size,
        "token_frequency_statistics": {
            "total_tokens_counted": total_tokens,
            "distinct_tokens_used": len(token_freq_counter),
            "vocab_utilization_pct": round(len(token_freq_counter) / vocab_size * 100, 2),
            "top_50_most_frequent_tokens": top_50_tokens
        },
        "average_characters_per_token": chars_per_token,
        "fertility_subwords_per_word": fertility,
        "unknown_token_statistics": {
            "total_unk_tokens": total_unk,
            "unk_rate_percentage": unk_rate,
            "byte_fallback_active": True
        },
        "tokenization_examples": tokenization_samples
    }

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hindi SentencePiece Tokenizer")
    parser.add_argument("--model", type=str, default="hindi/tokenizer/hindi_sp.model", help="Path to trained .model")
    parser.add_argument("--test_file", type=str, default="hindi/data/test.jsonl", help="Held-out test JSONL")
    parser.add_argument("--output", type=str, default="hindi/tokenizer/eval_metrics.json", help="Output metrics JSON")
    args = parser.parse_args()

    metrics = evaluate_tokenizer(os.path.abspath(args.model), os.path.abspath(args.test_file))
    
    out_path = os.path.abspath(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Evaluation metrics saved to {out_path}")
    logger.info(f"Summary -> Vocab: {metrics['vocabulary_size']}, Fertility: {metrics['fertility_subwords_per_word']}, UNK: {metrics['unknown_token_statistics']['unk_rate_percentage']}%, Chars/Token: {metrics['average_characters_per_token']}")

if __name__ == "__main__":
    main()
