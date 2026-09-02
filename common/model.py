"""
Decoder-Only Causal Transformer Language Model from First Principles in PyTorch (~25M Parameters)
================================================================================================
Implemented strictly using PyTorch primitive layers (nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout).
NO nn.Transformer*, NO HuggingFace model classes, NO pre-built attention blocks.
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """
    Hyperparameter configuration for the ~25M parameter causal Transformer LM.
    """
    vocab_size: int = 16000       # Standalone SentencePiece BPE vocabulary from Phase 1
    d_model: int = 512            # Embedding & hidden representation dimension
    n_layer: int = 6              # Number of Transformer blocks
    n_head: int = 8               # Number of parallel attention heads (d_k = 512 / 8 = 64)
    d_ff: int = 2048              # Position-wise feed-forward inner dimension (4 * d_model)
    max_seq_len: int = 256        # Maximum context sequence length (tokens)
    dropout: float = 0.1          # Dropout probability (applied to embeddings, attention, residual)
    tie_weights: bool = True      # Tie token embedding and output LM head weights
    bias: bool = True             # Use bias in linear projections and LayerNorm

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**d)


class TokenAndPositionalEmbedding(nn.Module):
    """
    Input Representation Stage:
    1. Token Embedding: Maps token IDs -> dense vectors in R^(vocab_size x d_model).
    2. Positional Embedding: Learned absolute positional embeddings in R^(max_seq_len x d_model).
       Self-attention is permutation-invariant, so positional coordinates must be explicitly injected.
       Learned absolute embeddings are robust, easy to inspect, and constrain the model's sequence length
       strictly to max_seq_len (tokens beyond max_seq_len cannot be indexed without extrapolation).
    3. Element-wise addition + embedding dropout.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.max_seq_len = config.max_seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (Batch_size B, Sequence_length T)
        Returns:
            Tensor of shape (Batch_size B, Sequence_length T, d_model)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max sequence length {self.max_seq_len}"

        # Position indices: [0, 1, 2, ..., T-1]
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)

        tok_emb = self.token_embed(input_ids)      # (B, T, d_model)
        pos_emb = self.pos_embed(positions)        # (T, d_model), broadcast to (B, T, d_model)

        x = self.dropout(tok_emb + pos_emb)        # (B, T, d_model)
        return x


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention implemented from first principles.
    
    Operations:
    1. Linear projections for Query (Q), Key (K), and Value (V).
    2. Explicit reshape and transpose: (B, T, d_model) -> (B, h, T, d_k).
    3. Scaled Dot-Product Attention:
       Attention(Q, K, V) = Softmax( (Q K^T) / sqrt(d_k) + M ) V
       - Scaling by 1 / sqrt(d_k) prevents dot-product values from growing excessively large
         in high dimensions (which would push softmax into regions of vanishingly small gradients).
    4. Additive Causal Mask (M):
       Upper-triangular matrix with 0 on allowed positions (j <= i) and -inf on future positions (j > i),
       ensuring position t can strictly attend only to positions <= t.
    5. Concatenation and output projection W_O: (B, h, T, d_k) -> (B, T, d_model).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0, f"d_model ({config.d_model}) must be divisible by n_head ({config.n_head})"
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_k = config.d_model // config.n_head  # Head dimension (64)

        # Q, K, V projections as separate linear layers
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Additive causal mask buffer: lower triangular is 0, strictly upper triangular is -inf
        mask = torch.triu(torch.full((config.max_seq_len, config.max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (B, T, d_model)
            return_attention: If True, also returns attention weight matrix of shape (B, h, T, T)
        Returns:
            out: Projected context representations of shape (B, T, d_model)
            attn_weights: Optional attention weights of shape (B, h, T, T)
        """
        B, T, C = x.shape

        # 1. Project Q, K, V: shape (B, T, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into h heads: (B, T, d_model) -> (B, T, h, d_k) -> transpose to (B, h, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product: (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. Apply Additive Causal Mask: position t attends only to positions <= t
        # Slice mask up to current sequence length T
        scores = scores + self.causal_mask[:T, :T]

        # 5. Softmax over key dimension (dim=-1)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # 6. Context aggregation: (B, h, T, T) @ (B, h, T, d_k) -> (B, h, T, d_k)
        out = torch.matmul(attn_weights_dropped, v)

        # 7. Concatenate heads: transpose to (B, T, h, d_k) -> contiguous -> view (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Output projection and residual dropout
        out = self.resid_dropout(self.out_proj(out))

        if return_attention:
            return out, attn_weights
        return out, None


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    Two linear layers with an inner expansion dimension (d_ff = 4 * d_model = 2048) and GELU non-linearity.
    
    Architecture:
    Linear(d_model, d_ff) -> GELU() -> Dropout -> Linear(d_ff, d_model) -> Dropout
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, T, d_ff) -> GELU -> (B, T, d_model)
        h = self.act(self.fc1(x))
        h = self.dropout1(h)
        out = self.fc2(h)
        out = self.dropout2(out)
        return out


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block:
    Pre-norm applies LayerNorm BEFORE the sublayer (attention / FFN), providing an unobstructed
    gradient highway through the residual stream. This ensures stable training for multi-layer stacks
    without requiring fragile learning-rate warmup tuning.
    
    Formulation:
    x = x + Dropout(Attention(LayerNorm_1(x)))
    x = x + Dropout(FFN(LayerNorm_2(x)))
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LN Self-Attention Sublayer + Residual connection
        normed_x = self.ln1(x)
        attn_out, attn_weights = self.attn(normed_x, return_attention=return_attention)
        x = x + attn_out

        # Pre-LN Feed-Forward Sublayer + Residual connection
        x = x + self.ffn(self.ln2(x))

        return x, attn_weights


class TransformerLM(nn.Module):
    """
    Decoder-Only Causal Language Model (~25M Parameters) from Primitives.
    
    Pipeline:
    Input IDs (B, T)
      -> Token & Positional Embeddings (B, T, d_model)
      -> Stack of N Transformer Blocks with Pre-LN and Causal Masking
      -> Final LayerNorm
      -> Linear Projection Head to Vocabulary Logits (B, T, vocab_size)
    
    Weight Tying:
    Input token embeddings and output projection head weights are tied.
    This saves ~8.2M redundant parameters (16,000 * 512 = 8,192,000 floats)
    while improving generalization by ensuring input and output semantic representations
    share the same embedding space.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. Input embeddings
        self.embedding = TokenAndPositionalEmbedding(config)

        # 2. Stack of N Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # 3. Final normalization layer
        self.ln_f = nn.LayerNorm(config.d_model)

        # 4. Output projection head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 5. Weight tying
        if config.tie_weights:
            self.lm_head.weight = self.embedding.token_embed.weight

        # 6. Weight initialization
        self.apply(self._init_weights)

        # Scaled initialization for residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of token IDs of shape (B, T)
            targets: Optional ground-truth next-token targets of shape (B, T)
            return_attentions: If True, returns attention weight matrices for all layers
            
        Returns:
            logits: Next-token logits of shape (B, T, vocab_size)
            loss: Cross-entropy loss scalar (if targets provided, else None)
            attentions: List of attention weight tensors [(B, h, T, T)] per layer (if return_attentions=True)
        """
        B, T = input_ids.shape

        # 1. Embed tokens and positions
        x = self.embedding(input_ids)  # (B, T, d_model)

        # 2. Forward through Transformer blocks
        all_attentions = [] if return_attentions else None
        for block in self.blocks:
            x, attn_w = block(x, return_attention=return_attentions)
            if return_attentions and attn_w is not None:
                all_attentions.append(attn_w)

        # 3. Final layer normalization
        x = self.ln_f(x)  # (B, T, d_model)

        # 4. Project to vocabulary logits: (B, T, vocab_size)
        logits = self.lm_head(x)

        # 5. Compute causal cross-entropy loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten predictions and targets: (B * T, vocab_size) and (B * T)
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

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
        """
        Autoregressive text generation with greedy decoding (temperature=0.0 or <= 1e-6)
        or stochastic temperature sampling.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if it exceeds max sequence length
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            # Forward pass: get logits for the last token position
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature <= 1e-6:
                # Greedy decoding: select token with maximum logit
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Temperature scaling
                logits = logits / temperature

                # Optional Top-K truncation
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Sample from categorical distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append sampled token to sequence
            idx = torch.cat((idx, next_token), dim=1)

            # Early termination if EOS encountered
            if eos_id is not None and (next_token == eos_id).all():
                break

        return idx


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Computes exact parameter counts for reporting and rubric compliance.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Non-embedding parameters: excludes token and positional embeddings
    non_embedding_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "embed" not in n and "lm_head" not in n
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_embedding_parameters": non_embedding_params
    }


if __name__ == "__main__":
    # Quick sanity check
    cfg = ModelConfig()
    model = TransformerLM(cfg)
    counts = count_parameters(model)
    print("=" * 60)
    print("TransformerLM Architecture Initialized (~25M Parameters):")
    for k, v in counts.items():
        print(f"  {k}: {v:,} ({v / 1e6:.2f}M)")
    print("=" * 60)
    
    # Test forward pass with dummy batch
    dummy_x = torch.randint(0, cfg.vocab_size, (2, 32))
    dummy_y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss, _ = model(dummy_x, targets=dummy_y)
    print(f"Logits shape: {logits.shape} (Expected: 2, 32, {cfg.vocab_size})")
    print(f"Loss scalar:  {loss.item():.4f}")
