"""
Reasoning Supervised Finetuning (SFT) Engine
============================================
Finetunes a pretrained ~25M causal Transformer LM on comparative and transitive
reasoning datasets using prompt-masked cross-entropy loss.
"""

import os
import sys
import time
import math
import json
import random
import csv
import argparse
from typing import Optional, Tuple, List, Dict, Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import matplotlib.pyplot as plt

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))
from common.model import TransformerLM, ModelConfig
from common.eval_reasoning import evaluate_reasoning

device = "cuda" if torch.cuda.is_available() else "cpu"

class SFTReasoningDataset:
    def __init__(self, jsonl_path: str, sp_model: spm.SentencePieceProcessor, max_len: int = 128):
        self.examples = []
        pad_id = sp_model.pad_id()
        eos_id = sp_model.eos_id()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                prompt = d["prompt"]
                full_text = d["text"]
                target = d["target"].strip()

                prompt_ids = sp_model.encode(prompt, out_type=int)
                full_ids = sp_model.encode(full_text, out_type=int) + [eos_id]

                if len(full_ids) > max_len:
                    full_ids = full_ids[:max_len]

                input_ids = full_ids[:-1]
                target_ids = list(full_ids[1:])

                # Mask out prompt tokens in target with -100
                prompt_len = len(prompt_ids)
                for i in range(min(prompt_len - 1, len(target_ids))):
                    target_ids[i] = -100

                self.examples.append({
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                    "prompt": prompt,
                    "target": target,
                    "task_type": d.get("task_type", "general")
                })

    def __len__(self):
        return len(self.examples)

    def get_batches(self, batch_size: int, shuffle: bool = True, device: str = "cpu") -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        indices = list(range(len(self.examples)))
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_inputs = [self.examples[idx]["input_ids"] for idx in batch_idx]
            batch_targets = [self.examples[idx]["target_ids"] for idx in batch_idx]

            max_len = max(len(b) for b in batch_inputs)
            padded_inputs = [b + [0] * (max_len - len(b)) for b in batch_inputs]
            padded_targets = [b + [-100] * (max_len - len(b)) for b in batch_targets]

            x = torch.tensor(padded_inputs, dtype=torch.long, device=device)
            y = torch.tensor(padded_targets, dtype=torch.long, device=device)
            yield x, y

def quick_eval_accuracy(model: TransformerLM, sp: spm.SentencePieceProcessor, dataset: SFTReasoningDataset, device: str, max_check: int = 50) -> float:
    model.eval()
    correct = 0
    sub_examples = dataset.examples[:max_check]
    total = len(sub_examples)
    eos_id = sp.eos_id()

    for item in sub_examples:
        prompt_ids = sp.encode(item["prompt"], out_type=int)
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out_ids = model.generate(inp, max_new_tokens=8, temperature=0.0, eos_id=eos_id)
        
        pred_tokens = out_ids[0, len(prompt_ids):].tolist()
        pred_text = sp.decode(pred_tokens).strip().strip("।,:.!? \t\n")
        first_word = pred_text.split()[0] if pred_text.split() else ""

        target = item["target"]
        if (first_word == target) or (target in pred_text.split()[:2]) or (pred_text.startswith(target)):
            correct += 1

    return round(correct / total * 100, 2) if total > 0 else 0.0

def train_sft(
    checkpoint_path: str,
    config_path: str,
    tokenizer_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    output_dir: str,
    max_steps: int = 1000,
    batch_size: int = 16,
    lr: float = 5e-5,
    save_prefix: str = "model_finetuned"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)
    model = TransformerLM(config).to(device)

    print(f"Loading pretrained weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print("Pretrained weights successfully loaded!")

    train_ds = SFTReasoningDataset(train_path, sp)
    val_ds = SFTReasoningDataset(val_path, sp)
    test_ds = SFTReasoningDataset(test_path, sp)
    print(f"Loaded datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    print("\nMeasuring Pretrained Baseline Test Accuracy...")
    base_acc = quick_eval_accuracy(model, sp, test_ds, device, max_check=50)
    print(f"Pretrained Zero-Shot Accuracy: {base_acc:.2f}%\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup_steps = 100

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    log_csv_path = os.path.join(output_dir, "finetune_log.csv")
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "lr", "train_loss", "val_loss", "val_acc"])

    step = 0
    best_val_acc = 0.0
    running_loss = 0.0
    start_time = time.time()

    steps_hist, train_loss_hist, val_loss_hist, val_acc_hist = [], [], [], []
    model.train()

    print(f"Training for {max_steps} steps (batch_size={batch_size}, lr={lr:.2e})...")
    while step < max_steps:
        for x, y in train_ds.get_batches(batch_size=batch_size, shuffle=True, device=device):
            if step >= max_steps:
                break

            current_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.zero_grad()
            _, loss, _ = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % 20 == 0:
                print(f"Step {step:4d}/{max_steps} | Loss: {running_loss/20:.4f} | LR: {current_lr:.2e}")
                running_loss = 0.0

            # Periodic evaluation
            if step % 100 == 0 or step == max_steps:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vx, vy in val_ds.get_batches(batch_size=batch_size, shuffle=False, device=device):
                        _, vloss, _ = model(vx, targets=vy)
                        val_losses.append(vloss.item())

                avg_vloss = sum(val_losses) / len(val_losses) if val_losses else 0.0
                val_acc = quick_eval_accuracy(model, sp, val_ds, device, max_check=50)

                steps_hist.append(step)
                train_loss_hist.append(loss.item())
                val_loss_hist.append(avg_vloss)
                val_acc_hist.append(val_acc)

                csv_writer.writerow([step, f"{current_lr:.2e}", f"{loss.item():.4f}", f"{avg_vloss:.4f}", f"{val_acc:.2f}"])
                csv_file.flush()
                print(f"--> Checkpoint @ Step {step:4d}: Val Loss = {avg_vloss:.4f} | Val Accuracy = {val_acc:.2f}%")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_ckpt = os.path.join(ckpt_dir, f"{save_prefix}_best.pt")
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "val_accuracy": val_acc,
                        "config": config.to_dict()
                    }, best_ckpt)
                    print(f"    * Saved New Best Checkpoint -> {best_ckpt} ({val_acc:.2f}%)")

                model.train()

    csv_file.close()

    final_ckpt = os.path.join(ckpt_dir, f"{save_prefix}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config.to_dict()
    }, final_ckpt)
    print(f"\nFinal Finetuned Checkpoint Saved -> {final_ckpt}")

    # Full held-out evaluation
    print("\nEvaluating Final Finetuned Model Accuracy on Held-out Test Set...")
    final_test_eval = evaluate_reasoning(model, sp, test_path, max_samples=50)
    final_test_acc = final_test_eval["overall_accuracy"]

    print("=" * 60)
    print(f"Pretrained Zero-Shot Accuracy : {base_acc:.2f}%")
    print(f"Finetuned Test Accuracy       : {final_test_acc:.2f}%")
    print(f"Accuracy Improvement          : +{final_test_acc - base_acc:.2f}%")
    print("=" * 60)

    return {
        "base_accuracy": base_acc,
        "finetuned_accuracy": final_test_acc,
        "best_val_accuracy": best_val_acc,
        "final_checkpoint": final_ckpt,
        "training_history": {
            "steps": steps_hist,
            "train_loss": train_loss_hist,
            "val_loss": val_loss_hist,
            "val_accuracy": val_acc_hist
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Supervised Finetuning on Reasoning Tasks")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_prefix", type=str, default="model_finetuned")
    args = parser.parse_args()

    train_sft(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        train_path=args.train_data,
        val_path=args.val_data,
        test_path=args.test_data,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        save_prefix=args.save_prefix
    )

if __name__ == "__main__":
    main()
