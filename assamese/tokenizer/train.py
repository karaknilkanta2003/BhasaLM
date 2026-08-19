"""
Assamese SentencePiece Tokenizer Training Script
Trains a standalone BPE tokenizer for Assamese (Assamese script U+0980-U+09FF).
"""

import os
import sys
import argparse
import tempfile
import sentencepiece as spm

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from common.utils import get_logger, load_jsonl

logger = get_logger("assamese_sp_train")

def train_spm_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 16000,
    character_coverage: float = 0.9995
) -> None:
    """
    Trains SentencePiece BPE tokenizer on the Assamese training split.
    """
    logger.info(f"Loading training data from {input_file}...")
    docs = load_jsonl(input_file)
    logger.info(f"Loaded {len(docs)} Assamese documents for training tokenizer.")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as temp_f:
        temp_path = temp_f.name
        for d in docs:
            text = d.get("text", "").strip()
            if text:
                temp_f.write(text + "\n")

    total_chars = sum(len(d["text"]) for d in docs)
    logger.info(f"Written plain text to {temp_path} (Total chars: {total_chars})")
    
    target_vocab = vocab_size

    spm_args = (
        f"--input={temp_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={target_vocab} "
        f"--model_type=bpe "
        f"--character_coverage={character_coverage} "
        f"--byte_fallback=true "
        f"--pad_id=0 "
        f"--unk_id=1 "
        f"--bos_id=2 "
        f"--eos_id=3 "
        f"--pad_piece=<pad> "
        f"--unk_piece=<unk> "
        f"--bos_piece=<s> "
        f"--eos_piece=</s> "
        f"--normalization_rule_name=identity "
        f"--hard_vocab_limit=false"
    )

    logger.info(f"Executing SentencePiece trainer for Assamese with args:\n{spm_args}")
    try:
        spm.SentencePieceTrainer.train(spm_args)
        logger.info(f"SentencePiece training succeeded! Output: {model_prefix}.model, {model_prefix}.vocab")
    except Exception as e:
        logger.error(f"SentencePiece training failed: {e}")
        raise e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    parser = argparse.ArgumentParser(description="Train Assamese SentencePiece BPE Tokenizer")
    parser.add_argument("--input", type=str, default="assamese/data/train.jsonl", help="Training JSONL split")
    parser.add_argument("--output_dir", type=str, default="assamese/tokenizer", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=16000, help="Target vocabulary size")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    model_prefix = os.path.join(out_dir, "assamese_sp")

    train_spm_tokenizer(
        input_file=os.path.abspath(args.input),
        model_prefix=model_prefix,
        vocab_size=args.vocab_size
    )

if __name__ == "__main__":
    main()
