"""
Self-Contained Kaggle GPU Pretraining Pipeline: Hindi Model H WITHOUT Positional Embeddings
===========================================================================================
Course: CL3-410 Language Models and Architecture — Bonus Ablation
Author: Nilkanta Karak
Branch: bonus

Scientific Setup:
- Architecture: 6 Layers, 8 Heads, d_model=512, d_ff=2048, max_seq_len=256, vocab=16000 (~27.1M params)
- ABLATION: PureTokenEmbedding without positional embeddings (positions = None, pos_embed = None)
- Precision: PyTorch AMP FP16 on NVIDIA Tesla T4 GPU
- Optimization: AdamW (lr=6e-4, warmup=1000 steps, cosine decay to 6e-5, 25,000 steps)
- Batching: Batch size 64 * sequence length 256 = 16,384 tokens/step (approx. 410M tokens)
- Logging: Pure tabular CSV (step, tokens_seen, lr, train_loss, val_loss, val_ppl, speed)
- Evaluation: Validation loss and perplexity computed every 500 steps over 490k tokens
- Checkpoints: Atomic state_dict saved every 1,000 steps to checkpoints/model_h_no_pos_step_{step}.pt
"""

import os
import sys
import time
import math
import json
import random
import csv
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import matplotlib.pyplot as plt

# Indic script font support for plots
plt.rcParams['font.sans-serif'] = ['Nirmala UI', 'Segoe UI', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. Model Configuration & Architecture (Ablated: No Positional Embeddings)
# ==============================================================================

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


# ==============================================================================
# 2. Dynamic Batch Streaming from Phase 1 Clean Corpus
# ==============================================================================

def stream_tokenized_batches(
    file_path: str,
    sp: spm.SentencePieceProcessor,
    batch_size: int = 64,
    seq_len: int = 256,
    device: str = "cuda"
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    chunk_size = seq_len + 1
    eos_id = sp.eos_id()
    buffer: List[int] = []

    while True:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if line.startswith("{") and line.endswith("}"):
                        doc = json.loads(line)
                        text = doc.get("text", "")
                    else:
                        text = line

                    if not text:
                        continue

                    tokens = sp.encode(text, out_type=int)
                    if not tokens:
                        continue
                    tokens.append(eos_id)
                    buffer.extend(tokens)

                    while len(buffer) >= batch_size * chunk_size:
                        batch_tokens = buffer[:batch_size * chunk_size]
                        buffer = buffer[batch_size * chunk_size:]

                        tensor_chunk = torch.tensor(batch_tokens, dtype=torch.long, device=device).view(batch_size, chunk_size)
                        inputs = tensor_chunk[:, :-1].contiguous()
                        targets = tensor_chunk[:, 1:].contiguous()
                        yield inputs, targets
                except Exception:
                    continue


# ==============================================================================
# 3. Path Resolution & Environment Auto-Detection
# ==============================================================================

def find_data_paths():
    print("\n[Inspecting Filesystem for Phase 1 Data & Tokenizer]")
    if os.path.exists("/kaggle/input"):
        print("Searching /kaggle/input:")
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"  {root} -> {len(files)} files: {files[:5]}")

    train_path, val_path, sp_path = None, None, None
    search_roots = ["/kaggle/input", "hindi_output", "hindi", "."]

    for search_root in search_roots:
        if os.path.exists(search_root):
            for root, _, files in os.walk(search_root):
                for f in files:
                    if f == "train.jsonl" and not train_path:
                        train_path = os.path.join(root, f)
                    elif f == "val.jsonl" and not val_path:
                        val_path = os.path.join(root, f)
                    elif f == "hindi_sp.model" and not sp_path:
                        sp_path = os.path.join(root, f)

    # Fallbacks if text format
    if not train_path:
        for search_root in search_roots:
            if os.path.exists(search_root):
                for root, _, files in os.walk(search_root):
                    for f in files:
                        if f == "train.txt" and not train_path:
                            train_path = os.path.join(root, f)
                        elif f == "val.txt" and not val_path:
                            val_path = os.path.join(root, f)

    if not (train_path and val_path and sp_path):
        raise FileNotFoundError(
            f"Could not locate all required files. Found: train={train_path}, val={val_path}, sp={sp_path}"
        )

    return train_path, val_path, sp_path


# ==============================================================================
# 4. Main Pretraining Execution Loop
# ==============================================================================

def main():
    print("=" * 75)
    print("Pretraining Ablated Hindi Model H (NO Positional Embeddings)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU: {gpu_name} (Capability {cap[0]}.{cap[1]}, {total_mem:.2f} GB VRAM)")
        if cap[0] < 7:
            print("[WARNING] GPU capability < 7.0 is incompatible with current PyTorch CUDA kernels.")
            device = "cpu"
        else:
            print(f"Accelerating on compatible GPU: {gpu_name} with PyTorch AMP FP16!")
    else:
        print("Hardware Device: CPU")

    output_dir = "/kaggle/working" if os.path.exists("/kaggle") else "./results_hindi_ablation_no_pos"
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Locate Datasets and Tokenizer
    train_path, val_path, sp_path = find_data_paths()
    print(f"Train Dataset:    {train_path}")
    print(f"Val Dataset:      {val_path}")
    print(f"Tokenizer Model:  {sp_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(sp_path)
    actual_vocab_size = sp.get_piece_size()
    print(f"SentencePiece Loaded: Vocab Size = {actual_vocab_size:,}")

    # 2. Model Initialization (NO Positional Embeddings)
    config = ModelConfigNoPos(
        vocab_size=actual_vocab_size,
        d_model=512,
        n_layer=6,
        n_head=8,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.1,
        tie_weights=True,
        use_pos_embed=False  # ABLATION ACTIVE
    )

    model = TransformerLMNoPos(config).to(device)
    param_counts = model.count_parameters()
    print(f"Ablated Model Parameters: {param_counts['total']:,} total, {param_counts['unique']:,} unique")
    print(f"Parameter Reduction vs Standard: -131,072 parameters (strictly NO positional embedding table)")

    # 3. Training Hyperparameters
    batch_size = 64
    seq_len = 256
    tokens_per_step = batch_size * seq_len  # 16,384 tokens / step
    max_steps = 25_000                      # 25,000 steps * 16,384 = ~409M tokens
    warmup_steps = 1000
    peak_lr = 6e-4
    min_lr = 6e-5
    grad_clip = 1.0
    eval_interval = 500
    save_interval = 1000

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        if step > max_steps:
            return min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (peak_lr - min_lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # CSV Logger
    log_csv_path = os.path.join(output_dir, "training_log.csv")
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "tokens_seen", "lr", "train_loss", "val_loss", "val_ppl", "tokens_per_sec"])
    csv_file.flush()

    steps_hist, train_loss_hist, val_loss_hist = [], [], []

    # Dataloaders
    train_loader = stream_tokenized_batches(train_path, sp, batch_size=batch_size, seq_len=seq_len, device=device)
    val_loader = stream_tokenized_batches(val_path, sp, batch_size=batch_size, seq_len=seq_len, device=device)

    print(f"\nBeginning Ablated Training: {max_steps:,} Steps | Batch: {batch_size} | Context: {seq_len} tokens/step")
    print("-" * 75)

    model.train()
    start_time = time.time()
    tokens_seen = 0

    for step in range(1, max_steps + 1):
        step_start = time.time()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = next(train_loader)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
            logits, loss, _ = model(x, targets=y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens_seen += tokens_per_step
        step_time = time.time() - step_start
        tok_per_sec = tokens_per_step / step_time

        # Periodic Console Progress (every 50 steps)
        if step % 50 == 0 or step == 1:
            print(f"Step {step:5d}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:.0f} tok/s | Tokens: {tokens_seen/1e6:.1f}M")

        # Periodic Validation (every eval_interval steps)
        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(30):  # Evaluate over 30 batches (491,520 tokens)
                    vx, vy = next(val_loader)
                    with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
                        _, v_loss, _ = model(vx, targets=vy)
                    val_losses.append(v_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
            print("*" * 75)
            print(f"--> EVAL STEP {step}: Val Loss = {avg_val_loss:.4f} | Val Perplexity = {val_ppl:.2f}")
            print("*" * 75)

            # Sample Generation Test
            sample_prompt = "महाभारत और रामायण"
            prompt_ids = torch.tensor([sp.encode(sample_prompt, out_type=int)], device=device)
            gen_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=50)
            gen_text = sp.decode(gen_ids[0].tolist())
            print(f"[Sample Gen @ Step {step}]: {gen_text}\n" + "-" * 75)

            # Write to CSV
            csv_writer.writerow([step, tokens_seen, f"{lr:.6e}", f"{loss.item():.4f}", f"{avg_val_loss:.4f}", f"{val_ppl:.2f}", f"{tok_per_sec:.0f}"])
            csv_file.flush()

            steps_hist.append(step)
            train_loss_hist.append(loss.item())
            val_loss_hist.append(avg_val_loss)

            model.train()

        # Periodic Checkpointing
        if step % save_interval == 0 or step == max_steps:
            ckpt_file = os.path.join(ckpt_dir, f"model_h_no_pos_step_{step}.pt")
            temp_file = f"{ckpt_file}.tmp"
            save_payload = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": loss.item(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.to_dict()
            }
            torch.save(save_payload, temp_file)
            os.replace(temp_file, ckpt_file)
            print(f"[Checkpoint Saved]: {ckpt_file}")

    csv_file.close()
    elapsed_hrs = (time.time() - start_time) / 3600
    print("=" * 75)
    print(f"Pretraining Completed Successfully! Total Elapsed Time: {elapsed_hrs:.2f} hours")
    print(f"Final Model Weights Saved: {ckpt_dir}")
    print("=" * 75)

    # Save final model checkpoint
    final_ckpt = os.path.join(ckpt_dir, "model_h_no_pos_final.pt")
    torch.save({
        "step": max_steps,
        "model_state_dict": model.state_dict(),
        "config": config.to_dict()
    }, final_ckpt)

    # Plot ablation training curve
    if steps_hist:
        plt.figure(figsize=(9, 5), dpi=300)
        plt.plot(steps_hist, train_loss_hist, label="Train Loss (No Pos Embed)", color="#d62728", linewidth=2)
        plt.plot(steps_hist, val_loss_hist, label="Val Loss (No Pos Embed)", color="#ff7f0e", linestyle="--", linewidth=2)
        plt.title("Ablation Pretraining: Hindi Model H without Positional Embeddings", fontsize=12, fontweight="bold")
        plt.xlabel("Step", fontsize=11)
        plt.ylabel("Cross-Entropy Loss", fontsize=11)
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        curve_path = os.path.join(fig_dir, "ablation_no_pos_loss_curve.png")
        plt.savefig(curve_path)
        plt.close()
        print(f"Saved ablation loss curve -> {curve_path}")


if __name__ == "__main__":
    main()
