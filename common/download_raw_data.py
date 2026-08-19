"""
Raw Full Dataset Downloader for Hindi and Assamese (~600M Raw Tokens Each)
Streams and saves 100% authentic, uncleaned raw text directly from the exact training sources.
"""

import os
import sys
import json
import gzip
import argparse
from datasets import load_dataset

HINDI_TARGET_RAW_TOKENS = 600_000_000
ASSAMESE_TARGET_RAW_TOKENS = 600_000_000

def download_raw_hindi(output_path: str, target_tokens: int = HINDI_TARGET_RAW_TOKENS, compress: bool = True):
    print("=" * 70)
    print(f"Downloading Full Raw Hindi Corpus (~{target_tokens:,} tokens) -> {output_path}")
    print("=" * 70)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sources = [
        ("wikimedia/wikipedia", "20231101.hi", "train", "raw_hindi_wikipedia"),
        ("ai4bharat/IndicCorpV2", "indiccorp_v2", "hin_Deva", "raw_indiccorp_hindi"),
        ("ai4bharat/sangraha", "verified", "hin", "raw_sangraha_verified_hindi")
    ]

    total_docs = 0
    total_words = 0
    total_est_tokens = 0
    open_fn = gzip.open if compress else open
    mode = "wt" if compress else "w"

    with open_fn(output_path, mode, encoding="utf-8") as f_out:
        for dataset_name, config_name, split_name, src_label in sources:
            if total_est_tokens >= target_tokens:
                break
            print(f"\n[Hindi] Streaming raw source: {dataset_name} ({config_name or 'default'})...")
            try:
                if config_name:
                    ds = load_dataset(dataset_name, config_name, split=split_name, streaming=True)
                else:
                    ds = load_dataset(dataset_name, split=split_name, streaming=True)

                for item in ds:
                    raw_text = str(item.get("text", "")).strip()
                    title = str(item.get("title", "")) if "title" in item else ""
                    if not raw_text:
                        continue

                    words = len(raw_text.split())
                    if words == 0:
                        continue
                    est_tok = int(words * 1.34)

                    record = {
                        "id": f"raw_hi_{total_docs + 1}",
                        "title": title,
                        "raw_uncleaned_text": raw_text,
                        "word_count": words,
                        "est_tokens": est_tok,
                        "source": src_label,
                        "language": "hi"
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_docs += 1
                    total_words += words
                    total_est_tokens += est_tok

                    if total_docs % 50_000 == 0:
                        size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        pct = round(total_est_tokens / target_tokens * 100, 2)
                        print(f"[Hindi Raw] {total_docs:,} docs | {total_words:,} words | ~{total_est_tokens:,} tokens ({pct}%) | File Size: {size_mb:.1f} MB")

                    if total_est_tokens >= target_tokens:
                        print(f"\n[Hindi Raw Complete] Target reached: ~{total_est_tokens:,} tokens ({total_docs:,} docs).")
                        break
            except Exception as e:
                print(f"Error streaming {dataset_name}: {e}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSuccessfully saved {total_docs:,} RAW Hindi documents ({size_mb:.2f} MB, ~{total_est_tokens:,} tokens) to: {output_path}")

def download_raw_assamese(output_path: str, target_tokens: int = ASSAMESE_TARGET_RAW_TOKENS, compress: bool = True):
    print("=" * 70)
    print(f"Downloading Full Raw Assamese Corpus (~{target_tokens:,} tokens) -> {output_path}")
    print("=" * 70)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sources = [
        ("wikimedia/wikipedia", "20231101.as", "train", "raw_assamese_wikipedia"),
        ("ai4bharat/sangraha", "verified", "asm", "raw_sangraha_verified_asm"),
        ("ai4bharat/IndicCorpV2", "indiccorp_v2", "asm_Beng", "raw_indiccorp_asm"),
        ("ai4bharat/sangraha", "unverified", "asm", "raw_sangraha_web_asm"),
        ("Deepjyoti120/AssameseDataTrain", None, "train", "raw_deepjyoti_news_asm")
    ]

    total_docs = 0
    total_words = 0
    total_est_tokens = 0
    open_fn = gzip.open if compress else open
    mode = "wt" if compress else "w"

    with open_fn(output_path, mode, encoding="utf-8") as f_out:
        for dataset_name, config_name, split_name, src_label in sources:
            if total_est_tokens >= target_tokens:
                break
            print(f"\n[Assamese] Streaming raw source: {dataset_name} ({config_name or 'default'})...")
            try:
                if config_name:
                    ds = load_dataset(dataset_name, config_name, split=split_name, streaming=True)
                else:
                    ds = load_dataset(dataset_name, split=split_name, streaming=True)

                for item in ds:
                    raw_text = str(item.get("text", "")).strip()
                    title = str(item.get("title", "")) if "title" in item else ""
                    if not raw_text:
                        continue

                    words = len(raw_text.split())
                    if words == 0:
                        continue
                    est_tok = int(words * 1.62)

                    record = {
                        "id": f"raw_as_{total_docs + 1}",
                        "title": title,
                        "raw_uncleaned_text": raw_text,
                        "word_count": words,
                        "est_tokens": est_tok,
                        "source": src_label,
                        "language": "as"
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_docs += 1
                    total_words += words
                    total_est_tokens += est_tok

                    if total_docs % 50_000 == 0:
                        size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        pct = round(total_est_tokens / target_tokens * 100, 2)
                        print(f"[Assamese Raw] {total_docs:,} docs | {total_words:,} words | ~{total_est_tokens:,} tokens ({pct}%) | File Size: {size_mb:.1f} MB")

                    if total_est_tokens >= target_tokens:
                        print(f"\n[Assamese Raw Complete] Target reached: ~{total_est_tokens:,} tokens ({total_docs:,} docs).")
                        break
            except Exception as e:
                print(f"Error streaming {dataset_name}: {e}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSuccessfully saved {total_docs:,} RAW Assamese documents ({size_mb:.2f} MB, ~{total_est_tokens:,} tokens) to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Download 600M raw uncleaned datasets for Hindi and Assamese")
    parser.add_argument("--lang", choices=["hi", "as", "both"], default="both", help="Language to download")
    parser.add_argument("--out_dir", type=str, default="raw_data", help="Output directory")
    parser.add_argument("--tokens", type=int, default=600_000_000, help="Target raw tokens per language")
    parser.add_argument("--no_compress", action="store_true", help="Do not gzip compress")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    compress = not args.no_compress
    ext = ".jsonl.gz" if compress else ".jsonl"

    if args.lang in ["hi", "both"]:
        download_raw_hindi(os.path.join(out_dir, f"raw_hindi_600m{ext}"), target_tokens=args.tokens, compress=compress)
    if args.lang in ["as", "both"]:
        download_raw_assamese(os.path.join(out_dir, f"raw_assamese_600m{ext}"), target_tokens=args.tokens, compress=compress)

if __name__ == "__main__":
    main()
