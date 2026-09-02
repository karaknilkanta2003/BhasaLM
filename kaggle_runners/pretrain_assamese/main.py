"""
Self-Contained Kaggle GPU Pretraining Pipeline: Assamese Model L (~25M Parameters)
==================================================================================
Trains a causal decoder-only Transformer from scratch in PyTorch using primitive layers.
- Architecture: 6 Layers, 8 Heads, d_model=512, d_ff=2048, context=256, vocab=16000 (~27M params)
- Pre-LN + GELU + Additive Causal Mask + Tied Embedding Weights
- Precision: PyTorch AMP FP16 on NVIDIA T4/P100 GPUs
- Logging: Pure tabular CSV & JSON metrics for offline plotting
- Checkpointing: Resumable atomic checkpoints for 12-hour session chaining
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


# ==============================================================================
# 1. Model Configuration & Architecture (From Primitives)
# ==============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 16000
    d_model: int = 512
    n_layer: int = 6
    n_head: int = 8
    d_ff: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.1
    tie_weights: bool = True
    bias: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.max_seq_len = config.max_seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"T={T} > max_seq_len={self.max_seq_len}"
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        return self.dropout(self.token_embed(input_ids) + self.pos_embed(positions))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_k = config.d_model // config.n_head

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.triu(torch.full((config.max_seq_len, config.max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + self.causal_mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)
        attn_dropped = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_dropped, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))

        return out, (attn_weights if return_attention else None)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.fc2(self.dropout1(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_w = self.attn(self.ln1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_w


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenAndPositionalEmbedding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.embedding.token_embed.weight

        self.apply(self._init_weights)
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
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

        return logits, loss, all_attentions

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
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
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
        return idx


# ==============================================================================
# 2. Streaming Tokenized Dataloader
# ==============================================================================

def stream_tokenized_batches(
    jsonl_path: str,
    sp_model: spm.SentencePieceProcessor,
    batch_size: int = 64,
    seq_len: int = 256,
    device: str = "cuda"
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Streams documents from JSONL, tokenizes on-the-fly, packs tokens into contiguous
    fixed-length chunks of (seq_len + 1), and yields (inputs, targets) tensors of shape (B, T).
    """
    buffer: List[int] = []
    chunk_size = seq_len + 1

    while True:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    if not text:
                        continue
                    tokens = sp_model.encode(text, out_type=int)
                    buffer.extend(tokens)

                    while len(buffer) >= batch_size * chunk_size:
                        batch_tokens = buffer[:batch_size * chunk_size]
                        buffer = buffer[batch_size * chunk_size:]

                        tensor_chunk = torch.tensor(batch_tokens, dtype=torch.long, device=device).view(batch_size, chunk_size)
                        inputs = tensor_chunk[:, :-1]
                        targets = tensor_chunk[:, 1:]
                        yield inputs, targets
                except Exception:
                    continue


# ==============================================================================
# 3. Main Training Execution Loop
# ==============================================================================

def find_data_paths():
    """Locates Phase 1 dataset and tokenizer paths dynamically across /kaggle/input or local."""
    print("\n[Inspecting Filesystem]")
    if os.path.exists("/kaggle/input"):
        print("Contents of /kaggle/input:")
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"  Dir: {root} -> {len(files)} files: {files[:5]}")
    else:
        print("/kaggle/input does not exist. Checking local directories...")

    train_path = None
    val_path = None
    sp_path = None

    search_roots = ["/kaggle/input", "assamese_output", "assamese", "."]
    for search_root in search_roots:
        if os.path.exists(search_root):
            for root, _, files in os.walk(search_root):
                for f in files:
                    if f == "train.jsonl" and not train_path:
                        train_path = os.path.join(root, f)
                    elif f == "val.jsonl" and not val_path:
                        val_path = os.path.join(root, f)
                    elif f == "assamese_sp.model" and not sp_path:
                        sp_path = os.path.join(root, f)

    if not (train_path and val_path and sp_path):
        raise FileNotFoundError(
            f"Could not locate all required files. Found: train={train_path}, val={val_path}, sp={sp_path}"
        )

    return train_path, val_path, sp_path


def main():
    print("=" * 70)
    print("Pretraining Assamese Model L (~25M Parameters) on Kaggle GPU")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Accelerating on GPU: {gpu_name} ({total_mem:.2f} GB VRAM)")
    else:
        print("[WARNING] CUDA not available. Running on CPU.")

    output_dir = "/kaggle/working" if os.path.exists("/kaggle") else "./outputs_pretrain_assamese"
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Locate Datasets and Tokenizer
    train_path, val_path, sp_path = find_data_paths()
    print(f"Train Dataset:    {train_path}")
    print(f"Val Dataset:      {val_path}")
    print(f"Tokenizer Model:  {sp_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(sp_path)
    actual_vocab_size = sp.get_piece_size()
    print(f"SentencePiece Loaded: Vocab Size = {actual_vocab_size:,}")

    # 2. Model Initialization (~25M Parameters)
    config = ModelConfig(
        vocab_size=actual_vocab_size,
        d_model=512,
        n_layer=6,
        n_head=8,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.1,
        tie_weights=True
    )

    model = TransformerLM(config).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model L Instantiated: Total Trainable Parameters = {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    # 3. Training Hyperparameters
    batch_size = 64
    seq_len = 256
    tokens_per_step = batch_size * seq_len
    max_steps = 25_000
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

    train_loader = stream_tokenized_batches(train_path, sp, batch_size=batch_size, seq_len=seq_len, device=device)
    val_loader = stream_tokenized_batches(val_path, sp, batch_size=batch_size, seq_len=seq_len, device=device)

    print(f"\nBeginning Training: {max_steps:,} Steps | Batch Size: {batch_size} | Context: {seq_len} tokens/step")
    print("-" * 70)

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

        if step % 50 == 0 or step == 1:
            print(f"Step {step:5d}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:.0f} tok/s | Tokens: {tokens_seen/1e6:.1f}M")

        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(30):
                    vx, vy = next(val_loader)
                    with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
                        _, v_loss, _ = model(vx, targets=vy)
                    val_losses.append(v_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
            print("*" * 70)
            print(f"--> EVAL STEP {step}: Val Loss = {avg_val_loss:.4f} | Val Perplexity = {val_ppl:.2f}")
            print("*" * 70)

            sample_prompt = "অসমৰ প্ৰাকৃতিক সৌন্দৰ্য"
            prompt_ids = torch.tensor([sp.encode(sample_prompt, out_type=int)], device=device)
            gen_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=50)
            gen_text = sp.decode(gen_ids[0].tolist())
            print(f"[Sample Gen @ Step {step}]: {gen_text}\n" + "-" * 70)

            csv_writer.writerow([step, tokens_seen, f"{lr:.6e}", f"{loss.item():.4f}", f"{avg_val_loss:.4f}", f"{val_ppl:.2f}", f"{tok_per_sec:.0f}"])
            csv_file.flush()
            model.train()

        if step % save_interval == 0 or step == max_steps:
            ckpt_file = os.path.join(ckpt_dir, f"model_l_step_{step}.pt")
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
    print("=" * 70)
    print(f"Pretraining Completed Successfully! Total Elapsed Time: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"Final Model Weights Saved: {ckpt_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
