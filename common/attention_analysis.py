"""
Attention Analysis Suite for Monolingual Transformer Language Models
====================================================================
Implements:
1. Attention weight extraction across all layers and heads.
2. Attention entropy per head/layer (measures focus vs. dispersion).
3. Mean attention distance per head/layer (measures local vs. long-range context).
4. Local vs. content-based head classification.
5. High-resolution attention heatmap generation for in-language sentences.
"""

import math
from typing import List, Dict, Tuple, Optional
import torch


def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes average Shannon entropy across query positions for a single attention head:
    H = - 1/T * sum_{i=1}^T sum_{j=1}^i A_{i,j} * log(A_{i,j} + eps)
    
    Interpretation:
    - High entropy (~uniform distribution) indicates diffuse attention (broad contextual smoothing).
    - Low entropy indicates sharply focused attention (e.g. specific keywords, immediate previous token).
    
    Args:
        attn_weights: Tensor of shape (T, T) for a single head.
    """
    T = attn_weights.size(0)
    # Mask out upper triangle
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=attn_weights.device))
    valid_weights = torch.where(mask, attn_weights, torch.zeros_like(attn_weights))

    # Entropy per query row i: - sum_{j <= i} p * log(p)
    log_weights = torch.log(valid_weights + eps)
    row_entropy = -(valid_weights * log_weights).sum(dim=-1) # (T,)

    return row_entropy.mean().item()


def compute_mean_attention_distance(attn_weights: torch.Tensor) -> float:
    """
    Computes mean attention distance along the sequence:
    Dist = 1/T * sum_{i=1}^T sum_{j=1}^i A_{i,j} * (i - j)
    
    Interpretation:
    - Small distance (< 2.0 tokens) indicates strictly local/positional heads (attending to token t-1 or t).
    - Large distance (> 5.0 tokens) indicates long-range / content-based heads (attending to distant nouns/subjects).
    
    Args:
        attn_weights: Tensor of shape (T, T) for a single head.
    """
    T = attn_weights.size(0)
    # Distance matrix: D_{i, j} = max(i - j, 0)
    i_indices = torch.arange(T, device=attn_weights.device).unsqueeze(1)
    j_indices = torch.arange(T, device=attn_weights.device).unsqueeze(0)
    dist_matrix = (i_indices - j_indices).clamp(min=0).float()

    # Average distance weighted by attention
    weighted_dist = (attn_weights * dist_matrix).sum(dim=-1) # (T,)
    return weighted_dist.mean().item()


def analyze_layer_attentions(
    layer_attentions: List[torch.Tensor]
) -> List[Dict[str, Any]]:
    """
    Analyzes attention matrices across all layers and heads.
    
    Args:
        layer_attentions: List of N tensors, each of shape (B, n_head, T, T).
    Returns:
        List of head statistics dictionaries.
    """
    summary = []
    # Take first item in batch
    for layer_idx, layer_tensor in enumerate(layer_attentions):
        layer_tensor = layer_tensor[0] # (n_head, T, T)
        n_head = layer_tensor.size(0)

        for head_idx in range(n_head):
            head_weights = layer_tensor[head_idx] # (T, T)
            entropy = compute_attention_entropy(head_weights)
            distance = compute_mean_attention_distance(head_weights)

            # Classify head role
            if distance < 2.0:
                head_type = "Local / Positional"
            elif entropy < 1.5:
                head_type = "Specialized / Focused"
            else:
                head_type = "Broad / Content-Based"

            summary.append({
                "layer": layer_idx + 1,
                "head": head_idx + 1,
                "entropy": round(entropy, 4),
                "mean_distance": round(distance, 2),
                "head_type": head_type
            })

    return summary
