"""
Assamese Phase 1 Master Pipeline Runner
Runs the full end-to-end data collection, cleaning, splitting, tokenizer training, and evaluation for Assamese.
"""

import os
import sys
import subprocess

def run_command(cmd: str) -> None:
    print(f"\n[RUNNING]: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[ERROR]: Command failed with exit code {res.returncode}: {cmd}")
        sys.exit(res.returncode)

def main():
    print("=" * 60)
    print("Starting Phase 1 Pipeline for Assamese (Model L)")
    print("=" * 60)

    # 1. Data Collection
    run_command(f"{sys.executable} assamese/data/collect.py --num_docs 1500")
    run_command(f"{sys.executable} assamese/data/collect_manual.py --num_docs 800")

    # 2. Cleaning & Deduplication
    run_command(f"{sys.executable} assamese/data/clean.py")

    # 3. Splitting & Corpus Statistics
    run_command(f"{sys.executable} assamese/data/split.py")

    # 4. Tokenizer Training (SentencePiece BPE 16k)
    run_command(f"{sys.executable} assamese/tokenizer/train.py --vocab_size 16000")

    # 5. Tokenizer Evaluation
    run_command(f"{sys.executable} assamese/tokenizer/evaluate.py")

    print("\n" + "=" * 60)
    print("Assamese Phase 1 Pipeline Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
