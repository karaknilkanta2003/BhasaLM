"""
Ablation Architecture: Decoder-Only Causal Transformer WITHOUT Positional Embeddings
=====================================================================================
Course: CL3-410 Language Models and Architecture — Bonus Ablation
Author: Nilkanta Karak
Branch: bonus

Scientific Rationale:
In a standard Transformer, token representations are augmented with positional encodings:
    x_t = e_token(w_t) + e_pos(t)
Without positional encodings (e_pos = 0), dot-product attention is permutation-symmetric.
The only structural inductive bias remaining is the causal triangular mask M_causal,
rendering the model a "causal bag-of-words" that cannot differentiate relative or
absolute token ordering within historical context.

Parameter Count:
Standard Model H: 27,238,400 parameters
Ablated Model H (No PE): 27,107,328 parameters (-131,072 parameters, exactly 256 * 512 removed)
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfigNoPos:
    vocab_size: int = 16000
    d_model: int = 512
    n_layer: int = 6
    n_head: int = 8
    d_ff: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.1
    tie_weights: bool = True
    bias: bool = True
    use_pos_embed: bool = False  # Strictly False for ablation

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfigNoPos":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PureTokenEmbedding(nn.Module):
    """
    Token-only embedding layer with NO positional representations.
    Maps token IDs to d_model dimensional vectors with residual dropout.
    No positional lookup table is instantiated or added.
    """
    def __init__(self, config: ModelConfigNoPos):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.max_seq_len = config.max_seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}"
        # Pure token embeddings: zero positional information injected
        x = self.token_embed(input_ids)
        return self.dropout(x)


class CausalSelfAttentionNoPos(nn.Module):
    """
    Multi-Head Causal Self-Attention block.
    Computes scaled dot-product attention over causal lower-triangular mask.
    Without positional embeddings, attention weights over past tokens depend purely
    on semantic token-token correlations, causing permutation invariance over history.
    """
    def __init__(self, config: ModelConfigNoPos):
        super().__init__()
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_k = config.d_model // config.n_head
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask (strictly lower triangular)
        mask = torch.triu(torch.full((config.max_seq_len, config.max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape

        # Linear projections & split into h heads: (B, h, T, d_k)
        q = self.q_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # Attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + self.causal_mask[:T, :T]

        attn_weights = F.softmax(scores, dim=-1)
        attn_dropped = self.attn_dropout(attn_weights)

        # Context aggregation
        out = torch.matmul(attn_dropped, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))

        return out, (attn_weights if return_attention else None)


class FeedForwardNoPos(nn.Module):
    """
    Position-wise Feed-Forward Network:
    Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model) -> Dropout
    """
    def __init__(self, config: ModelConfigNoPos):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.dropout1(h)
        out = self.fc2(h)
        out = self.dropout2(out)
        return out


class TransformerBlockNoPos(nn.Module):
    """
    Pre-LayerNorm Transformer Block without positional inductive bias:
    x = x + Attention(LN_1(x))
    x = x + FFN(LN_2(x))
    """
    def __init__(self, config: ModelConfigNoPos):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttentionNoPos(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForwardNoPos(config)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_w = self.attn(self.ln1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_w


class TransformerLMNoPos(nn.Module):
    """
    Complete Ablated Decoder-Only Causal Language Model (~27.1M parameters).
    Strictly omits positional embeddings to study the degradation of language modeling,
    grammatical coherence, and attention mechanisms in the absence of sequence position.
    """
    def __init__(self, config: ModelConfigNoPos):
        super().__init__()
        self.config = config
        self.embedding = PureTokenEmbedding(config)
        self.blocks = nn.ModuleList([TransformerBlockNoPos(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: lm_head shares weights with token_embed
        if config.tie_weights:
            self.lm_head.weight = self.embedding.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tied = self.embedding.token_embed.weight.numel() if self.config.tie_weights else 0
        return {
            "total": total,
            "trainable": trainable,
            "unique": total - tied if self.config.tie_weights else total
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        B, T = input_ids.shape
        x = self.embedding(input_ids)

        all_attentions = [] if return_attentions else None
        for block in self.blocks:
            x, attn_w = block(x, return_attention=return_attentions)
            if return_attentions and attn_w is not None:
                all_attentions.append(attn_w)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss, all_attentions

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 1e-6:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break

        return idx
