"""
Post-Finetuning Attention Analysis Suite (Phase 3)
=================================================
Compares internal self-attention mechanics between Pretrained Base LM
and Finetuned Reasoning LM on comparative/transitive reasoning prompts:
1. Early (Layer 1) vs. Late (Layer 6) Heatmaps for Pretrained vs. Finetuned.
2. Attention Entropy shift (measures focus vs. dispersion).
3. Mean Attention Distance shift (measures local syntax vs. long-range premise binding).
"""

import os
import sys
import json
import torch
import sentencepiece as spm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))
from common.model import TransformerLM, ModelConfig
from common.attention_analysis import compute_attention_entropy, compute_mean_attention_distance

# Indic fonts
plt.rcParams['font.sans-serif'] = ['Nirmala UI', 'Segoe UI', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path: str, config_path: str) -> TransformerLM:
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)
    model = TransformerLM(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model

def plot_pretrain_vs_finetune_attention(
    model_pre: TransformerLM,
    model_ft: TransformerLM,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    lang_name: str,
    save_path: str
) -> Dict[str, Any]:
    """
    Plots a 2x2 comparative grid:
    - Row 1: Pretrained Layer 1 & Layer 6
    - Row 2: Finetuned Layer 1 & Layer 6
    """
    tokens = sp.encode_as_pieces(prompt)[:14]
    token_ids = sp.encode_as_ids(prompt)[:14]
    clean_tokens = [t.replace("\u2581", "_") for t in tokens]

    inp = torch.tensor([token_ids], dtype=torch.long, device=device)

    # 1. Pretrained Attention Forward
    with torch.no_grad():
        _, _, attn_pre = model_pre(inp, return_attentions=True)
    l1_pre = attn_pre[0][0, 0].cpu().numpy() # Head 0
    l6_pre = attn_pre[5][0, 0].cpu().numpy() # Head 0

    # 2. Finetuned Attention Forward
    with torch.no_grad():
        _, _, attn_ft = model_ft(inp, return_attentions=True)
    l1_ft = attn_ft[0][0, 0].cpu().numpy()
    l6_ft = attn_ft[5][0, 0].cpu().numpy()

    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), dpi=300)

    # Pretrained Layer 1
    sns.heatmap(l1_pre, xticklabels=clean_tokens, yticklabels=clean_tokens, cmap="Blues", ax=axes[0, 0], cbar_kws={"label": "Weight"})
    axes[0, 0].set_title(f"[Pretrained] {lang_name} Layer 1: Local / Positional", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Query Position", fontsize=10)

    # Pretrained Layer 6
    sns.heatmap(l6_pre, xticklabels=clean_tokens, yticklabels=clean_tokens, cmap="Purples", ax=axes[0, 1], cbar_kws={"label": "Weight"})
    axes[0, 1].set_title(f"[Pretrained] {lang_name} Layer 6: Semantic / Diffuse", fontsize=11, fontweight="bold")

    # Finetuned Layer 1
    sns.heatmap(l1_ft, xticklabels=clean_tokens, yticklabels=clean_tokens, cmap="GnBu", ax=axes[1, 0], cbar_kws={"label": "Weight"})
    axes[1, 0].set_title(f"[Finetuned] {lang_name} Layer 1: Positional + Entity Anchor", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Key Position (Attended)", fontsize=10)
    axes[1, 0].set_ylabel("Query Position", fontsize=10)

    # Finetuned Layer 6
    sns.heatmap(l6_ft, xticklabels=clean_tokens, yticklabels=clean_tokens, cmap="YlOrRd", ax=axes[1, 1], cbar_kws={"label": "Weight"})
    axes[1, 1].set_title(f"[Finetuned] {lang_name} Layer 6: Comparative Entity Binding", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Key Position (Attended)", fontsize=10)

    plt.suptitle(
        f"Reasoning Attention Dynamics: Pretrained vs. Finetuned ({lang_name})\nPrompt: \"{prompt[:45]}...\"",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comparative attention plot -> {save_path}")

    # Entropy & Distance metrics
    def get_stats(attn_list):
        l1_ent = compute_attention_entropy(attn_list[0][0, 0])
        l6_ent = compute_attention_entropy(attn_list[5][0, 0])
        l1_dist = compute_mean_attention_distance(attn_list[0][0, 0])
        l6_dist = compute_mean_attention_distance(attn_list[5][0, 0])
        return {
            "l1_entropy": round(l1_ent, 3),
            "l6_entropy": round(l6_ent, 3),
            "l1_distance": round(l1_dist, 2),
            "l6_distance": round(l6_dist, 2)
        }

    stats_pre = get_stats(attn_pre)
    stats_ft = get_stats(attn_ft)

    return {
        "language": lang_name,
        "prompt": prompt,
        "pretrained": stats_pre,
        "finetuned": stats_ft,
        "entropy_delta_l6": round(stats_ft["l6_entropy"] - stats_pre["l6_entropy"], 3),
        "distance_delta_l6": round(stats_ft["l6_distance"] - stats_pre["l6_distance"], 2)
    }
