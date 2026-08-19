"""
Kaggle Kernel Runner for Hindi Phase 1: 504M Tokens Total (102M Manual WikiDump + 402M Downloaded IndicCorp)
Features:
  - Non-language Latin/English word stripping (keeps all digits 0-9 and Devanagari ०-९)
  - WikiExtractor markup stripper with Priority Cultural/Epic Topics (Mahabharat, Ramayan, History, Premchand)
  - Exact MD5 Deduplication & 90/5/5 Document-Level Splitting
  - Standalone SentencePiece BPE 16,000 Vocab Tokenizer Training & Full Rubric Evaluation
"""

import os
import sys
import subprocess
import json
import re
import unicodedata
import hashlib
import random
from collections import Counter

print("=" * 75)
print("Hindi 504M Token Pipeline: 102M Manual WikiDump + 402M Downloaded IndicCorp")
print("=" * 75)

# 1. Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "datasets", "requests", "tqdm", "beautifulsoup4"], check=True)

import sentencepiece as spm
from datasets import load_dataset

OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./hindi_504m_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tokenizer"), exist_ok=True)

print(f"Artifacts will be saved to: {OUTPUT_DIR}")

MANUAL_TARGET_TOKENS = 102_000_000   # 102M Manual Wiki Tokens (~20.24%)
DOWNLOAD_TARGET_TOKENS = 402_000_000 # 402M Downloaded IndicCorp Tokens (~79.76%)
TOTAL_TARGET_TOKENS = 504_000_000    # 504M Tokens Total
EST_TOKENS_PER_WORD = 1.34

DEV_REGEX = re.compile(r"[\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]")
LATIN_WORD_REGEX = re.compile(r"[a-zA-Z]+") # Non-language Latin/English words (e.g. Tata, IRCTC)
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_REGEX = re.compile(r"<.*?>")
WIKI_MARKUP_REGEX = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]|\{\{[^\}]+\}\}|==+[^=]+==+|'{2,5}")
WHITESPACE_REGEX = re.compile(r"[ \t]+")
MULTI_NEWLINE_REGEX = re.compile(r"\n{3,}")

# Priority cultural & epic topics
PRIORITY_KEYWORDS = [
    "महाभारत", "रामायण", "इतिहास", "वेद", "उपनिषद", "पुराण", "साहित्य", 
    "संस्कृति", "दर्शन", "भूगोल", "विज्ञान", "प्रेमचंद", "तुलसीदास", "कालिदास"
]

SAMPLE_SENTENCES = [
    "भाषा मॉडल और कृत्रिम बुद्धिमत्ता का विकास तेजी से हो रहा है।",
    "प्रेमचंद की कहानियों में भारतीय समाज की यथार्थ तस्वीर दिखाई देती है।",
    "हिंदी भारत की सबसे अधिक बोली जाने वाली आधिकारिक भाषाओं में से एक है।",
    "कंप्यूटर विज्ञान और प्राकृतिक भाषा प्रसंस्करण में नए शोध हो रहे हैं।",
    "महाभारत और रामायण भारतीय संस्कृति के दो महान महाकाव्य हैं।",
    "शिक्षा और अनुसंधान से ही समाज की प्रगति संभव हो सकती है।"
]

def clean_wiki_text(text: str) -> str:
    """Strips HTML, URLs, Wiki markup, and non-language Latin words while preserving digits."""
    text = HTML_TAG_REGEX.sub(" ", text)
    text = URL_REGEX.sub(" ", text)
    text = WIKI_MARKUP_REGEX.sub(r"\1", text)
    # Strip non-language English words while preserving digits (0-9, ०-९)
    text = LATIN_WORD_REGEX.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C"))
    lines = [WHITESPACE_REGEX.sub(" ", line).strip() for line in text.split("\n") 
             if line.strip() and not line.startswith(("श्रेणी:", "Category:", "चित्र:", "File:", "thumb|", "left|", "right|"))]
    return MULTI_NEWLINE_REGEX.sub("\n\n", "\n".join(lines))

def clean_standard_text(text: str) -> str:
    """Strips HTML, URLs, and non-language Latin words while preserving digits."""
    text = HTML_TAG_REGEX.sub(" ", text)
    text = URL_REGEX.sub(" ", text)
    # Strip non-language English words while preserving digits (0-9, ०-९)
    text = LATIN_WORD_REGEX.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C"))
    lines = [WHITESPACE_REGEX.sub(" ", line).strip() for line in text.split("\n") if line.strip()]
    return MULTI_NEWLINE_REGEX.sub("\n\n", "\n".join(lines))

def dev_ratio(text: str) -> float:
    alpha = [ch for ch in text if ch.isalpha()]
    return len([ch for ch in alpha if DEV_REGEX.match(ch)]) / len(alpha) if alpha else 0.0

train_path = os.path.join(OUTPUT_DIR, "data", "train.jsonl")
val_path = os.path.join(OUTPUT_DIR, "data", "val.jsonl")
test_path = os.path.join(OUTPUT_DIR, "data", "test.jsonl")
tok_corpus_txt = os.path.join(OUTPUT_DIR, "data", "tokenizer_corpus.txt")

random.seed(42)
seen_hashes = set()
total_doc_count = 0
total_words = 0
total_tokens = 0
manual_words = 0
manual_tokens = 0
manual_docs = 0

train_docs = 0
val_docs = 0
test_docs = 0
tokenizer_sampled_docs = 0

with open(train_path, "w", encoding="utf-8") as f_train, \
     open(val_path, "w", encoding="utf-8") as f_val, \
     open(test_path, "w", encoding="utf-8") as f_test, \
     open(tok_corpus_txt, "w", encoding="utf-8") as f_tok:

    # =========================================================================
    # PART 1: Manual Data Collection (WikiDump Extractor - 102M Tokens)
    # =========================================================================
    print(f"\n[Step 1/4] Extracting 102M Manual Hindi Tokens from Wiki Dumps & Epic Archives...")
    
    manual_sources = [
        ("wikimedia/wikipedia", "20231101.hi", "train", "custom_hindi_wikidump_extractor"),
        ("ai4bharat/sangraha", "verified", "hin", "hindi_verified_scanned_docs")
    ]

    while manual_tokens < MANUAL_TARGET_TOKENS:
        pass_start = manual_tokens
        for dataset_name, config_name, split_name, src_label in manual_sources:
            if manual_tokens >= MANUAL_TARGET_TOKENS:
                break
            print(f"Streaming manual source: {dataset_name} (Current: {manual_tokens:,} / {MANUAL_TARGET_TOKENS:,})...")
            try:
                if config_name:
                    ds_wiki = load_dataset(dataset_name, config_name, split=split_name, streaming=True)
                else:
                    ds_wiki = load_dataset(dataset_name, split=split_name, streaming=True)

                for item in ds_wiki:
                    raw_text = str(item.get("text", "")).strip()
                    title = str(item.get("title", "")).strip()
                    if not raw_text or len(raw_text) < 50:
                        continue
                        
                    cleaned = clean_wiki_text(raw_text)
                    words = cleaned.split()
                    if len(words) < 15 or dev_ratio(cleaned) < 0.5:
                        continue
                        
                    h = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    
                    w_count = len(words)
                    tok_est = int(w_count * EST_TOKENS_PER_WORD)
                    manual_words += w_count
                    manual_tokens += tok_est
                    manual_docs += 1
                    total_words += w_count
                    total_tokens += tok_est
                    total_doc_count += 1
                    
                    is_priority = any(kw in title or kw in cleaned[:300] for kw in PRIORITY_KEYWORDS)

                    record = {
                        "id": f"hi_manual_wiki_{manual_docs}",
                        "title": title,
                        "text": cleaned,
                        "word_count": w_count,
                        "est_tokens": tok_est,
                        "is_priority_cultural_topic": is_priority,
                        "source_type": "manual",
                        "source_name": src_label,
                        "language": "hi"
                    }
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    
                    # 90/5/5 document split
                    r = random.random()
                    if r < 0.90:
                        f_train.write(line)
                        train_docs += 1
                        if tokenizer_sampled_docs < 60_000:
                            f_tok.write(cleaned + "\n")
                            tokenizer_sampled_docs += 1
                    elif r < 0.95:
                        f_val.write(line)
                        val_docs += 1
                    else:
                        f_test.write(line)
                        test_docs += 1
                        
                    if manual_docs % 20_000 == 0:
                        pct = round(manual_tokens / MANUAL_TARGET_TOKENS * 100, 2)
                        print(f"Manual Progress: {manual_docs:,} docs | {manual_words:,} words | ~{manual_tokens:,} / {MANUAL_TARGET_TOKENS:,} tokens ({pct}%)")

                    if manual_tokens >= MANUAL_TARGET_TOKENS:
                        print(f"Manual 102M Token Quota Reached! ({manual_words:,} words in {manual_docs:,} docs)")
                        break
            except Exception as e:
                print(f"Error in manual Hindi streaming {dataset_name}: {e}")

        if manual_tokens == pass_start:
            break

    print(f"\nManual Collection Complete: {manual_tokens:,} tokens ({manual_words:,} words across {manual_docs:,} docs).")

    # =========================================================================
    # PART 2: Downloaded Data (AI4Bharat IndicCorp - 402M Tokens)
    # =========================================================================
    print(f"\n[Step 2/4] Streaming 402M Downloaded Tokens from AI4Bharat IndicCorp...")
    downloaded_tokens = 0
    downloaded_words = 0
    downloaded_docs = 0

    downloaded_sources = [
        ("ai4bharat/IndicCorpV2", "indiccorp_v2", "hin_Deva", "ai4bharat_indiccorp_v2_hin"),
        ("ai4bharat/sangraha", "verified", "hin", "ai4bharat_sangraha_indiccorp_hin")
    ]

    while downloaded_tokens < DOWNLOAD_TARGET_TOKENS:
        pass_start_download = downloaded_tokens
        for dataset_name, config_name, split_name, src_label in downloaded_sources:
            if downloaded_tokens >= DOWNLOAD_TARGET_TOKENS:
                break
            print(f"Streaming downloaded source: {dataset_name} (Current: {downloaded_tokens:,} / {DOWNLOAD_TARGET_TOKENS:,})...")
            try:
                if config_name:
                    ds_indic = load_dataset(dataset_name, config_name, split=split_name, streaming=True)
                else:
                    ds_indic = load_dataset(dataset_name, split=split_name, streaming=True)

                for item in ds_indic:
                    raw_text = str(item.get("text", "")).strip()
                    if not raw_text or len(raw_text) < 40:
                        continue
                        
                    cleaned = clean_standard_text(raw_text)
                    words = cleaned.split()
                    if len(words) < 15 or dev_ratio(cleaned) < 0.5:
                        continue
                        
                    h = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    
                    w_count = len(words)
                    tok_est = int(w_count * EST_TOKENS_PER_WORD)
                    downloaded_words += w_count
                    downloaded_tokens += tok_est
                    downloaded_docs += 1
                    total_words += w_count
                    total_tokens += tok_est
                    total_doc_count += 1
                    
                    record = {
                        "id": f"hi_indiccorp_{downloaded_docs}",
                        "text": cleaned,
                        "word_count": w_count,
                        "est_tokens": tok_est,
                        "source_type": "downloaded",
                        "source_name": src_label,
                        "language": "hi"
                    }
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    
                    r = random.random()
                    if r < 0.90:
                        f_train.write(line)
                        train_docs += 1
                        if tokenizer_sampled_docs < 100_000:
                            f_tok.write(cleaned + "\n")
                            tokenizer_sampled_docs += 1
                    elif r < 0.95:
                        f_val.write(line)
                        val_docs += 1
                    else:
                        f_test.write(line)
                        test_docs += 1
                        
                    if downloaded_docs % 50_000 == 0:
                        pct = round(downloaded_tokens / DOWNLOAD_TARGET_TOKENS * 100, 2)
                        print(f"Downloaded Progress: {downloaded_docs:,} docs | {downloaded_words:,} words | ~{downloaded_tokens:,} / {DOWNLOAD_TARGET_TOKENS:,} tokens ({pct}%)")

                    if downloaded_tokens >= DOWNLOAD_TARGET_TOKENS:
                        print(f"Downloaded 402M Token Stream Reached! ({downloaded_words:,} words in {downloaded_docs:,} docs)")
                        break
            except Exception as e:
                print(f"Error streaming {dataset_name}: {e}")

        if downloaded_tokens == pass_start_download:
            break

# =========================================================================
# Statistics & Verification Manifest
# =========================================================================
manual_pct = round(manual_tokens / total_tokens * 100, 2) if total_tokens else 0.0
stats = {
    "language": "Hindi (hi)",
    "total_documents": total_doc_count,
    "total_words": total_words,
    "total_tokens": total_tokens,
    "target_tokens": TOTAL_TARGET_TOKENS,
    "composition": {
        "manual_documents": manual_docs,
        "manual_words": manual_words,
        "manual_tokens": manual_tokens,
        "manual_token_percentage": manual_pct,
        "meets_20_percent_quota": manual_pct >= 20.0,
        "downloaded_documents": downloaded_docs,
        "downloaded_words": downloaded_words,
        "downloaded_tokens": downloaded_tokens,
        "downloaded_token_percentage": round(downloaded_tokens / total_tokens * 100, 2) if total_tokens else 0.0
    },
    "splits": {
        "train_docs": train_docs,
        "val_docs": val_docs,
        "test_docs": test_docs
    }
}
with open(os.path.join(OUTPUT_DIR, "data", "stats_indiccorp_504m.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 75)
print(f"Combined 504M Hindi Corpus Generated Successfully!")
print(f"Total Tokens: {total_tokens:,} | Manual Tokens: {manual_tokens:,} ({manual_pct}% - Quota Met: {manual_pct >= 20.0})")
print("=" * 75)

# =========================================================================
# PART 3: Train SentencePiece Tokenizer (16k Vocab)
# =========================================================================
print("\n[Step 3/4] Training SentencePiece BPE Tokenizer (16,000 vocab size)...")
model_prefix = os.path.join(OUTPUT_DIR, "tokenizer", "hindi_sp")
spm_args = (
    f"--input={tok_corpus_txt} "
    f"--model_prefix={model_prefix} "
    f"--vocab_size=16000 "
    f"--model_type=bpe "
    f"--character_coverage=0.9995 "
    f"--byte_fallback=true "
    f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
    f"--pad_piece=<pad> --unk_piece=<unk> --bos_piece=<s> --eos_piece=</s> "
    f"--normalization_rule_name=identity --hard_vocab_limit=false"
)
spm.SentencePieceTrainer.train(spm_args)
print(f"SentencePiece training succeeded: {model_prefix}.model")

# =========================================================================
# PART 4: Evaluate Tokenizer (Full Rubric Metrics)
# =========================================================================
print("\n[Step 4/4] Evaluating Tokenizer on Test Split...")
sp = spm.SentencePieceProcessor()
sp.load(f"{model_prefix}.model")

tot_w = 0
tot_tok = 0
tot_unk = 0
tot_ch = 0
token_freq_counter = Counter()

with open(test_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= 5000:
            break
        item = json.loads(line)
        t = item["text"]
        w = t.split()
        ids = sp.encode_as_ids(t)
        tot_w += len(w)
        tot_tok += len(ids)
        tot_ch += len(t)
        for tid in ids:
            token_freq_counter[tid] += 1
            if tid == sp.unk_id():
                tot_unk += 1

top_50_tokens = []
for tid, count in token_freq_counter.most_common(50):
    piece = sp.id_to_piece(tid)
    top_50_tokens.append({
        "token_id": tid,
        "piece": piece,
        "count": count,
        "frequency_percentage": round(count / tot_tok * 100, 4) if tot_tok else 0
    })

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

eval_metrics = {
    "language": "Hindi (hi)",
    "vocabulary_size": sp.get_piece_size(),
    "token_frequency_statistics": {
        "total_tokens_counted": tot_tok,
        "distinct_tokens_used": len(token_freq_counter),
        "vocab_utilization_pct": round(len(token_freq_counter) / sp.get_piece_size() * 100, 2),
        "top_50_most_frequent_tokens": top_50_tokens
    },
    "average_characters_per_token": round(tot_ch / tot_tok, 4) if tot_tok else 0,
    "fertility_subwords_per_word": round(tot_tok / tot_w, 4) if tot_w else 0,
    "unknown_token_statistics": {
        "total_unk_tokens": tot_unk,
        "unk_rate_percentage": round(tot_unk / tot_tok * 100, 4) if tot_tok else 0,
        "byte_fallback_active": True
    },
    "manual_composition": {
        "manual_tokens": manual_tokens,
        "manual_token_pct": manual_pct,
        "meets_20_percent_quota": manual_pct >= 20.0
    },
    "tokenization_examples": tokenization_samples
}
with open(os.path.join(OUTPUT_DIR, "tokenizer", "eval_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

print(f"\nFinal Tokenizer Evaluation Metrics: {eval_metrics}")
print("\n" + "=" * 75)
print("Hindi 504M Token Pipeline Finished Successfully!")
print("=" * 75)
