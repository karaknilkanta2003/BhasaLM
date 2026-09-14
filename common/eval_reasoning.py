"""
Reasoning Evaluation Suite (Phase 3)
===================================
Evaluates language models on synthetic comparative and transitive reasoning benchmarks.
Computes:
1. Exact Match (EM) Accuracy
2. Per-task breakdown:
   - Transitive Ordering (A > B and B > C => Extrema)
   - Direct Quantitative Comparisons (Heights, Ages, Prices, Quantities)
   - Multi-Hop Inequality Verification (Yes / No)
3. Qualitative Success / Failure Case Logging
"""

import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any, Tuple
import torch
import sentencepiece as spm

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))
from common.model import TransformerLM, ModelConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint_and_model(checkpoint_path: str, config_path: str) -> Tuple[TransformerLM, ModelConfig]:
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)
    model = TransformerLM(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle state_dict key variations
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config

def clean_answer_text(text: str) -> str:
    """Extracts the immediate predicted entity/word and strips punctuation."""
    # Remove leading whitespace and punctuation
    text = text.strip().strip("।,:.!? \t\n")
    # Take first word or phrase before newline or punctuation
    parts = re.split(r"[\n।,:.!?]", text)
    if parts and parts[0].strip():
        return parts[0].strip().split()[0] if parts[0].strip().split() else ""
    return ""

def evaluate_reasoning(
    model: TransformerLM,
    sp: spm.SentencePieceProcessor,
    test_jsonl: str,
    max_samples: int = 500,
    max_new_tokens: int = 8
) -> Dict[str, Any]:
    """
    Evaluates exact match accuracy across all test questions.
    """
    model.eval()
    with open(test_jsonl, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if max_samples and len(items) > max_samples:
        items = items[:max_samples]

    total = len(items)
    correct = 0
    task_stats: Dict[str, Dict[str, int]] = {}
    examples: List[Dict[str, Any]] = []

    eos_id = sp.eos_id()

    for idx, item in enumerate(items):
        prompt = item["prompt"]
        target = item["target"].strip()
        task_type = item.get("task_type", "general")

        if task_type not in task_stats:
            task_stats[task_type] = {"total": 0, "correct": 0}
        task_stats[task_type]["total"] += 1

        prompt_ids = sp.encode(prompt, out_type=int)
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out_ids = model.generate(
                inp,
                max_new_tokens=max_new_tokens,
                temperature=0.0, # Pure Greedy Decoding
                eos_id=eos_id
            )

        gen_tokens = out_ids[0, len(prompt_ids):].tolist()
        raw_pred = sp.decode(gen_tokens).strip()
        cleaned_pred = clean_answer_text(raw_pred)

        # Check match: exact match or target entity prefix
        is_match = (cleaned_pred == target) or (target in raw_pred.split()[:2]) or (raw_pred.startswith(target))

        if is_match:
            correct += 1
            task_stats[task_type]["correct"] += 1

        # Keep a few qualitative examples
        if len(examples) < 15:
            examples.append({
                "id": item.get("id", f"sample_{idx}"),
                "task_type": task_type,
                "prompt": prompt,
                "target": target,
                "raw_prediction": raw_pred,
                "cleaned_prediction": cleaned_pred,
                "is_correct": is_match
            })

    overall_accuracy = round((correct / total * 100), 2) if total > 0 else 0.0

    task_breakdown = {}
    for t_name, t_data in task_stats.items():
        t_acc = round(t_data["correct"] / t_data["total"] * 100, 2) if t_data["total"] > 0 else 0.0
        task_breakdown[t_name] = {
            "total": t_data["total"],
            "correct": t_data["correct"],
            "accuracy": t_acc
        }

    return {
        "total_evaluated": total,
        "overall_correct": correct,
        "overall_accuracy": overall_accuracy,
        "task_breakdown": task_breakdown,
        "sample_cases": examples
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Reasoning Accuracy on Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config", type=str, required=True, help="Path to model config.json")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to sentencepiece .model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--max_samples", type=int, default=100, help="Max test samples to evaluate (default 100 for fast eval, 0 for all)")
    parser.add_argument("--output_json", type=str, default="", help="Optional JSON save path")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_checkpoint_and_model(args.checkpoint, args.config)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"Loaded Tokenizer: {args.tokenizer} (Vocab: {len(sp)})")

    limit = None if args.max_samples <= 0 else args.max_samples
    results = evaluate_reasoning(model, sp, args.test_data, max_samples=limit)
    print("\n" + "=" * 60)
    print(f"Reasoning Evaluation Results for: {os.path.basename(args.checkpoint)}")
    print("=" * 60)
    print(f"Total Evaluated: {results['total_evaluated']}")
    print(f"Overall Accuracy: {results['overall_accuracy']}% ({results['overall_correct']}/{results['total_evaluated']})")
    print("\nTask Breakdown:")
    for t_name, t_metrics in results["task_breakdown"].items():
        print(f"  - {t_name:28s}: {t_metrics['accuracy']:6.2f}% ({t_metrics['correct']}/{t_metrics['total']})")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved detailed results -> {args.output_json}")

if __name__ == "__main__":
    main()
