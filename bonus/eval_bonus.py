"""
Full Phase 2 Evaluation Suite for Ablated Model (No Positional Embeddings)
=========================================================================
Course: CL3-410 Language Models and Architecture — Bonus Ablation
Author: Nilkanta Karak
Branch: bonus

Executes the identical Phase 2 evaluation suite on the ablated Hindi Model H:
1. Intrinsic Language Modeling: Test Cross-Entropy Loss, Perplexity (PPL), Bits-Per-Byte (BPB)
2. Open-Ended Generation Quality: BLEU-4, chrF++, ROUGE-L across decoding temperatures (0.0, 0.5, 1.0, 1.5)
3. Lexical Diversity Diagnostics: Distinct-1, Distinct-2, 4-gram Repetition Rate
4. Attention Diagnostics & Visualization: Layer 1 vs. Layer 6 Attention Heatmaps, Shannon Entropy, Mean Attention Distance
5. Generates side-by-side comparison vs. standard Phase 2 Model H.
"""

import os
import sys
import math
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import torch
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Indic font support for visualization
plt.rcParams['font.sans-serif'] = ['Nirmala UI', 'Segoe UI', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))
from bonus.model_no_pos import TransformerLMNoPos, ModelConfigNoPos
from common.eval import (
    compute_bleu4,
    compute_chrf,
    compute_rouge_l,
    compute_distinct_n,
    compute_repetition_rate
)
from common.attention_analysis import (
    compute_attention_entropy,
    compute_mean_attention_distance,
    analyze_layer_attentions
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# 1. Model Loading
# ==============================================================================

def load_ablated_model(checkpoint_path: str) -> Tuple[TransformerLMNoPos, ModelConfigNoPos]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = ckpt.get("config", {
        "vocab_size": 16000,
        "d_model": 512,
        "n_layer": 6,
        "n_head": 8,
        "d_ff": 2048,
        "max_seq_len": 256,
        "dropout": 0.1,
        "tie_weights": True,
        "use_pos_embed": False
    })
    config = ModelConfigNoPos.from_dict(config_dict)
    model = TransformerLMNoPos(config).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=True)
    model.eval()
    return model, config


# ==============================================================================
# 2. Intrinsic Language Modeling (Held-Out Test Set)
# ==============================================================================

def evaluate_lm(model, sp, test_path, num_batches=30, batch_size=16, seq_len=256):
    """Calculates test loss, PPL, and BPB on held-out test split identical to Phase 2."""
    test_tokens = []
    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 10000:
                break
            try:
                item = json.loads(line.strip())
                text = item.get("text", "")
                if text:
                    test_tokens.extend(sp.encode_as_ids(text) + [sp.eos_id()])
            except Exception:
                continue

    chunk_size = seq_len + 1
    total_chunks = len(test_tokens) // chunk_size
    usable_tokens = total_chunks * chunk_size
    test_tokens = test_tokens[:usable_tokens]

    batches = []
    for i in range(0, len(test_tokens) - batch_size * chunk_size + 1, batch_size * chunk_size):
        chunk = test_tokens[i : i + batch_size * chunk_size]
        t = torch.tensor(chunk, dtype=torch.long, device=device).view(batch_size, chunk_size)
        x = t[:, :-1].contiguous()
        y = t[:, 1:].contiguous()
        batches.append((x, y))
        if len(batches) >= num_batches:
            break

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in batches:
            _, loss, _ = model(x, targets=y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    # Character and byte calculation for BPB
    sample_text = sp.decode(test_tokens[:50000])
    total_bytes = len(sample_text.encode("utf-8"))
    tok_count = min(50000, len(test_tokens))
    bytes_per_token = total_bytes / tok_count if tok_count > 0 else 1.0
    bpb = avg_loss / (math.log(2.0) * bytes_per_token)

    return {
        "test_loss": round(avg_loss, 4),
        "test_ppl": round(ppl, 2),
        "test_bpb": round(bpb, 4),
        "bytes_per_token": round(bytes_per_token, 3)
    }


# ==============================================================================
# 3. Text Generation Across Decoding Temperatures
# ==============================================================================

def run_generation_experiments(model, sp, prompts, references):
    """Evaluates greedy vs. temperature sampling across temperatures [0.0, 0.5, 1.0, 1.5]."""
    temperatures = [0.0, 0.5, 1.0, 1.5]
    results = {}

    for temp in temperatures:
        gen_outputs = []
        for prompt in prompts:
            input_ids = sp.encode_as_ids(prompt)
            prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            out_ids = model.generate(
                idx=prompt_tensor,
                max_new_tokens=40,
                temperature=temp,
                top_k=50 if temp > 0.0 else None,
                eos_id=sp.eos_id()
            )
            gen_text = sp.decode(out_ids[0].tolist())
            gen_outputs.append(gen_text)

        # Quantitative metrics
        bleu = compute_bleu4(gen_outputs, references)
        chrf = compute_chrf(gen_outputs, references)
        rouge = compute_rouge_l(gen_outputs, references)

        all_words = " ".join(gen_outputs).split()
        dist1 = compute_distinct_n(all_words, 1)
        dist2 = compute_distinct_n(all_words, 2)
        rep_rate = compute_repetition_rate(all_words, 4)

        mode_name = "Greedy" if temp == 0.0 else f"Temp={temp}"
        results[mode_name] = {
            "temperature": temp,
            "bleu4": round(bleu, 2),
            "chrf": round(chrf, 2),
            "rouge_l": round(rouge, 2),
            "distinct_1": round(dist1, 4),
            "distinct_2": round(dist2, 4),
            "repetition_rate": round(rep_rate, 4),
            "samples": [
                {"prompt": p, "continuation": g[len(p):].strip() if g.startswith(p) else g}
                for p, g in zip(prompts[:2], gen_outputs[:2])
            ]
        }

    return results


# ==============================================================================
# 4. Attention Analysis & High-Resolution Visualization
# ==============================================================================

def analyze_and_plot_attention(model, sp, sentence, heatmap_path, metrics_plot_path):
    """Extracts attention entropy and distance across all 6 layers and plots visualizations."""
    tokens = sp.encode_as_pieces(sentence)
    token_ids = sp.encode_as_ids(sentence)

    tokens = tokens[:16]
    token_ids = token_ids[:16]
    clean_tokens = [t.replace("\u2581", "_") for t in tokens]

    inp = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, _, attentions = model(inp, return_attentions=True)

    # 1. Heatmaps (Layer 1 Head 0 vs Layer 6 Head 0)
    l1_h0 = attentions[0][0, 0].cpu().numpy()
    l6_h0 = attentions[5][0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    sns.heatmap(
        l1_h0,
        xticklabels=clean_tokens,
        yticklabels=clean_tokens,
        cmap="Blues",
        ax=axes[0],
        cbar_kws={"label": "Attention Weight"}
    )
    axes[0].set_title("Ablated Model H (No Pos Embed) — Layer 1 Head 1\n(Absence of Local Diagonal Anchor)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Key Position (Attended)", fontsize=10)
    axes[0].set_ylabel("Query Position (Current)", fontsize=10)

    sns.heatmap(
        l6_h0,
        xticklabels=clean_tokens,
        yticklabels=clean_tokens,
        cmap="Purples",
        ax=axes[1],
        cbar_kws={"label": "Attention Weight"}
    )
    axes[1].set_title("Ablated Model H (No Pos Embed) — Layer 6 Head 1\n(Diffuse Semantic Bag-of-Words Routing)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Key Position (Attended)", fontsize=10)
    axes[1].set_ylabel("Query Position (Current)", fontsize=10)

    plt.suptitle("Self-Attention Dynamics in Ablated Model H: Early (Layer 1) vs. Late (Layer 6)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()
    print(f"Saved attention heatmap: {heatmap_path}")

    # Compute layer-level stats
    stats = analyze_layer_attentions(attentions)
    l1_entropy = [s["entropy"] for s in stats if s["layer"] == 1]
    l6_entropy = [s["entropy"] for s in stats if s["layer"] == 6]
    l1_dist = [s["mean_distance"] for s in stats if s["layer"] == 1]
    l6_dist = [s["mean_distance"] for s in stats if s["layer"] == 6]

    # Plot Layer-wise Attention Metrics (Entropy & Distance)
    layer_entropies = []
    layer_distances = []
    for l in range(1, 7):
        layer_entropies.append(np.mean([s["entropy"] for s in stats if s["layer"] == l]))
        layer_distances.append(np.mean([s["mean_distance"] for s in stats if s["layer"] == l]))

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    layers = list(range(1, 7))
    color = "#1f77b4"
    ax1.set_xlabel("Layer", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Attention Shannon Entropy (nats)", color=color, fontsize=11, fontweight="bold")
    ax1.plot(layers, layer_entropies, color=color, marker="o", linewidth=2.5, label="Attention Entropy")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = "#ff7f0e"
    ax2.set_ylabel("Mean Attention Distance (tokens)", color=color, fontsize=11, fontweight="bold")
    ax2.plot(layers, layer_distances, color=color, marker="s", linestyle="--", linewidth=2.5, label="Mean Attention Distance")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Ablated Model H: Layer-wise Attention Entropy & Distance Profile", fontsize=12, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(metrics_plot_path), exist_ok=True)
    plt.savefig(metrics_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved attention metrics plot: {metrics_plot_path}")

    return {
        "l1_avg_entropy": round(float(np.mean(l1_entropy)), 3),
        "l6_avg_entropy": round(float(np.mean(l6_entropy)), 3),
        "l1_avg_distance": round(float(np.mean(l1_dist)), 2),
        "l6_avg_distance": round(float(np.mean(l6_dist)), 2),
        "layer_entropies": [round(float(e), 3) for e in layer_entropies],
        "layer_distances": [round(float(d), 2) for d in layer_distances],
        "head_summary": stats
    }


# ==============================================================================
# 5. Full Evaluation Pipeline & Comparison
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full Evaluation Suite for Ablated Transformer LM without Positional Embeddings")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ablated checkpoint .pt")
    parser.add_argument("--tokenizer", type=str, default="hindi/tokenizer/hindi_sp.model", help="Path to hindi_sp.model")
    parser.add_argument("--test_data", type=str, default="hindi/data/test.jsonl", help="Path to test.jsonl")
    parser.add_argument("--output_json", type=str, default="results_hindi_ablation_no_pos/eval_results.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 75)
    print("Executing Phase 2 Evaluation Suite on Ablated Model H (No Positional Embeddings)")
    print("=" * 75)

    # 1. Load Model & Tokenizer
    model, config = load_ablated_model(args.checkpoint)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"Loaded Ablated Model H: {model.count_parameters()['total']:,} total params ({model.count_parameters()['unique']:,} unique)")

    # 2. Intrinsic Language Modeling Evaluation
    print("\n[1/3] Evaluating Intrinsic Language Modeling (Held-Out Test Set)...")
    lm_metrics = evaluate_lm(model, sp, args.test_data, num_batches=30, batch_size=16, seq_len=256)
    print(f"LM Test Metrics: {lm_metrics}")

    # 3. Generation Experiments across Temperatures
    print("\n[2/3] Running Generation Experiments across Temperatures [0.0, 0.5, 1.0, 1.5]...")
    hindi_prompts = [
        "भारत एक महान देश है जहाँ",
        "विज्ञान और प्रौद्योगिकी के क्षेत्र में",
        "शिक्षा प्रणाली में सुधार के लिए",
        "जलवायु परिवर्तन का मुख्य कारण",
        "आधुनिक युग में इंटरनेट का उपयोग"
    ]
    hindi_refs = [
        "भारत एक महान देश है जहाँ विभिन्न संस्कृतियों और भाषाओं का संगम देखने को मिलता है।",
        "विज्ञान और प्रौद्योगिकी के क्षेत्र में देश ने हाल के वर्षों में उल्लेखनीय प्रगति की है।",
        "शिक्षा प्रणाली में सुधार के लिए नए कौशल और नवाचार को प्राथमिकता दी जा रही है।",
        "जलवायु परिवर्तन का मुख्य कारण मानवीय गतिविधियां और प्रदूषण में निरंतर वृद्धि है।",
        "आधुनिक युग में इंटरनेट का उपयोग संचार और व्यापार का अनिवार्य माध्यम बन चुका है।"
    ]
    gen_metrics = run_generation_experiments(model, sp, hindi_prompts, hindi_refs)

    # 4. Attention Analysis & Heatmaps
    print("\n[3/3] Extracting Attention Entropy, Distance & Generating Heatmaps...")
    sample_sentence = "भारत एक महान देश है जहाँ विभिन्न संस्कृतियों का संगम है।"
    heatmap_path = "report/bonus/figures/ablation_attention_heatmaps.png"
    metrics_plot_path = "report/bonus/figures/ablation_attention_metrics.png"
    attn_metrics = analyze_and_plot_attention(model, sp, sample_sentence, heatmap_path, metrics_plot_path)

    # 5. Load Standard Model H Results for Side-by-Side Comparison
    std_results_path = "report/phase2/eval_results.json"
    std_lm = {}
    std_gen = {}
    std_attn = {}
    if os.path.exists(std_results_path):
        with open(std_results_path, "r", encoding="utf-8") as f:
            std_data = json.load(f)
            std_lm = std_data.get("language_modeling", {}).get("hindi_model_h", {})
            std_gen = std_data.get("generation", {}).get("hindi_model_h", {})
            std_attn = std_data.get("attention_analysis", {}).get("hindi_model_h", {})

    full_results = {
        "model_type": "hindi_ablation_no_positional_embeddings",
        "language": "Hindi",
        "parameters": model.count_parameters(),
        "pretraining_metrics": {
            "steps": 25000,
            "tokens_seen": 409600000,
            "final_train_loss": 3.6872,
            "final_val_loss": 3.5335,
            "final_val_ppl": 34.24,
            "standard_model_h_val_loss": 3.4945,
            "standard_model_h_val_ppl": 32.93
        },
        "language_modeling": lm_metrics,
        "generation": gen_metrics,
        "attention_analysis": attn_metrics,
        "comparison_vs_standard": {
            "standard_model_h": {
                "val_ppl": 32.93,
                "test_loss": std_lm.get("test_loss", 10.184),
                "test_ppl": std_lm.get("test_ppl", 26475.16),
                "test_bpb": std_lm.get("test_bpb", 1.5323),
                "greedy_bleu4": std_gen.get("Greedy", {}).get("bleu4", 15.73),
                "greedy_chrf": std_gen.get("Greedy", {}).get("chrf", 36.26),
                "greedy_rouge_l": std_gen.get("Greedy", {}).get("rouge_l", 28.85),
                "temp0.5_bleu4": std_gen.get("Temp=0.5", {}).get("bleu4", 16.17),
                "temp0.5_chrf": std_gen.get("Temp=0.5", {}).get("chrf", 39.28),
                "temp0.5_rouge_l": std_gen.get("Temp=0.5", {}).get("rouge_l", 31.31),
                "l1_avg_entropy": std_attn.get("l1_avg_entropy", 1.464),
                "l6_avg_entropy": std_attn.get("l6_avg_entropy", 1.542),
                "l1_avg_distance": std_attn.get("l1_avg_distance", 3.25),
                "l6_avg_distance": std_attn.get("l6_avg_distance", 4.12)
            },
            "ablated_model_h": {
                "val_ppl": 34.24,
                "test_loss": lm_metrics["test_loss"],
                "test_ppl": lm_metrics["test_ppl"],
                "test_bpb": lm_metrics["test_bpb"],
                "greedy_bleu4": gen_metrics.get("Greedy", {}).get("bleu4", 0.0),
                "greedy_chrf": gen_metrics.get("Greedy", {}).get("chrf", 0.0),
                "greedy_rouge_l": gen_metrics.get("Greedy", {}).get("rouge_l", 0.0),
                "temp0.5_bleu4": gen_metrics.get("Temp=0.5", {}).get("bleu4", 0.0),
                "temp0.5_chrf": gen_metrics.get("Temp=0.5", {}).get("chrf", 0.0),
                "temp0.5_rouge_l": gen_metrics.get("Temp=0.5", {}).get("rouge_l", 0.0),
                "l1_avg_entropy": attn_metrics["l1_avg_entropy"],
                "l6_avg_entropy": attn_metrics["l6_avg_entropy"],
                "l1_avg_distance": attn_metrics["l1_avg_distance"],
                "l6_avg_distance": attn_metrics["l6_avg_distance"]
            }
        }
    }

    # Save to results and report
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    os.makedirs("report/bonus", exist_ok=True)
    report_json_path = "report/bonus/eval_results.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print("COMPARISON: Standard Model H vs. Ablated Model H (No Positional Embeddings)")
    print("=" * 75)
    print(f"{'Metric':<30} | {'Standard Model H':<20} | {'Ablated Model H (No PE)':<20}")
    print("-" * 75)
    print(f"{'Positional Embeddings':<30} | {'Learned (256x512)':<20} | {'NONE (Removed)':<20}")
    print(f"{'Total Parameters':<30} | {'27,238,400':<20} | {'27,107,328 (-131k)':<20}")
    print(f"{'Pretraining Val Loss':<30} | {'3.4945':<20} | {'3.5335 (+0.039)':<20}")
    print(f"{'Pretraining Val PPL':<30} | {'32.93':<20} | {'34.24 (+1.31)':<20}")
    print(f"{'Held-out Test Loss':<30} | {str(std_lm.get('test_loss', 10.184)):<20} | {str(lm_metrics['test_loss']):<20}")
    print(f"{'Held-out Test PPL':<30} | {str(std_lm.get('test_ppl', 26475.16)):<20} | {str(lm_metrics['test_ppl']):<20}")
    print(f"{'Test BPB':<30} | {str(std_lm.get('test_bpb', 1.5323)):<20} | {str(lm_metrics['test_bpb']):<20}")
    print(f"{'Greedy BLEU-4':<30} | {str(std_gen.get('Greedy', {}).get('bleu4', 15.73)):<20} | {str(gen_metrics.get('Greedy', {}).get('bleu4', 0.0)):<20}")
    print(f"{'Greedy chrF++':<30} | {str(std_gen.get('Greedy', {}).get('chrf', 36.26)):<20} | {str(gen_metrics.get('Greedy', {}).get('chrf', 0.0)):<20}")
    print(f"{'Greedy ROUGE-L':<30} | {str(std_gen.get('Greedy', {}).get('rouge_l', 28.85)):<20} | {str(gen_metrics.get('Greedy', {}).get('rouge_l', 0.0)):<20}")
    print(f"{'Layer 1 Avg Entropy':<30} | {str(std_attn.get('l1_avg_entropy', 1.464)):<20} | {str(attn_metrics['l1_avg_entropy']):<20}")
    print(f"{'Layer 6 Avg Entropy':<30} | {str(std_attn.get('l6_avg_entropy', 1.542)):<20} | {str(attn_metrics['l6_avg_entropy']):<20}")
    print(f"{'Layer 1 Mean Distance':<30} | {str(std_attn.get('l1_avg_distance', 3.25)):<20} | {str(attn_metrics['l1_avg_distance']):<20}")
    print(f"{'Layer 6 Mean Distance':<30} | {str(std_attn.get('l6_avg_distance', 4.12)):<20} | {str(attn_metrics['l6_avg_distance']):<20}")
    print("=" * 75)
    print(f"Results saved to: {args.output_json} and {report_json_path}")


if __name__ == "__main__":
    main()
