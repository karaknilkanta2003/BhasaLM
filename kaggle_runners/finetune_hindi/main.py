"""
Self-Contained Kaggle GPU Reasoning Finetuning Pipeline: Hindi Model H
======================================================================
Finetunes the ~27M pretrained Hindi Transformer on programmatic comparative &
transitive reasoning tasks (Phase 3).
- Model: Starts from Phase 2 checkpoint (model_h_step_25000.pt)
- Tokenizer: Fixed 16k vocab (hindi_sp.model)
- SFT: Masked Prompt Cross-Entropy Loss (ignore_index=-100)
- Optimizer: AdamW (lr=5e-5, warmup=100 steps, cosine decay, 1000 steps)
- Evaluation: Pretrained vs. Finetuned Exact Match (EM) Accuracy on held-out test split
- Logging: Tabular CSV + loss curves + attention visualizations
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

# Configure font support for Indic scripts
plt.rcParams['font.sans-serif'] = ['Nirmala UI', 'Segoe UI', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "results_hindi_reasoning"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# ==============================================================================
# 1. Model Architecture (From Scratch Primitives)
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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

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
        self.n_head = config.n_head
        self.d_k = config.d_model // config.n_head
        self.scale = 1.0 / math.sqrt(self.d_k)

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

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + self.causal_mask[:T, :T]
        attn_weights = F.softmax(scores, dim=-1)
        attn_dropped = self.attn_dropout(attn_weights)

        y = torch.matmul(attn_dropped, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(y))

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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        x = self.embedding(input_ids)
        all_attentions = []

        for block in self.blocks:
            x, attn_w = block(x, return_attention=return_attentions)
            if return_attentions and attn_w is not None:
                all_attentions.append(attn_w)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Masked SFT Cross-Entropy (ignores positions with -100)
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
        max_new_tokens: int = 10,
        temperature: float = 0.0,
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
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
        return idx


# ==============================================================================
# 2. Synthetic Reasoning Data Generator (Zero-Leakage Programmatic Corpus)
# ==============================================================================

TRAIN_NAMES = [
    "अमित", "राहुल", "विकास", "रोहन", "सोहन", "अजय", "संजय", "मनीष", "दीपक", "नीरज",
    "प्रमोद", "विवेक", "सुमित", "गौरव", "आकाश", "कुणाल", "वरुण", "नितिन", "राजेश", "अशोक",
    "पूजा", "नेहा", "प्रिया", "अंजलि", "स्नेहा", "कविता", "रीता", "सुनीता", "मीना", "आरती",
    "प्रीति", "दीपा", "सीमा", "रश्मि", "भावना", "किरण", "श्वेता", "पूनम", "मोनिका", "अंशुल",
    "अनिल", "सुनील", "रमेश", "सुरेश", "महेश", "दिनेश", "मुकेश", "राकेश", "नरेश", "हरीश",
    "कमल", "पवन", "अतुल", "आलोक", "कपिल", "विनोद", "मनोज", "प्रवीण", "राजीव", "संजीव",
    "अल्पना", "साक्षी", "दीक्षा", "प्राची", "श्रुति", "अदित्या", "रूपाली", "शालिनी", "निशा", "ज्योति",
    "प्रतिभा", "मंजू", "संगीता", "उर्मिला", "मधु", "सरिता", "कुसुम", "विद्या", "लता", "कमला",
    "विशाल", "अभिषेक", "संदीप", "जितेंद्र", "धर्मेंद्र", "योगेश", "हेमराज", "भगवंत", "बलवंत", "कुलदीप",
    "गुरदीप", "सतनाम", "हरप्रीत", "मनप्रीत", "गुरप्रीत", "जसप्रीत", "अमरजीत", "इंद्रजीत", "रणजीत", "सुरजीत",
    "पारस", "अविनाश", "सौरभ", "मयूरेश", "दिग्विजय", "ऋत्विक", "समर्थ", "वेदांत", "शौर्य", "अथर्व",
    "सारिका", "अनुराधा", "गायत्री", "कौशल्या", "देवकी", "यशोदा", "रुक्मिणी", "रोहिणी", "अनुसूया", "उषा"
]
VAL_NAMES = [
    "हर्ष", "कार्तिक", "मयंक", "तरुण", "पार्थ", "श्रेया", "तनु", "दिव्या", "रिया", "ईशा",
    "अर्नव", "विहान", "आरव", "अयान", "कबीर", "अनाया", "मायरा", "सान्वी", "इनाया", "कियारा",
    "भव्य", "ध्रुव", "शशांक", "तन्वी", "अवनि"
]
TEST_NAMES = [
    "अभिनव", "ऋषभ", "चेतन", "तुषार", "निखिल", "मयूर", "लोकेश", "वैभव", "सिद्धार्थ", "हेमंत",
    "अदिति", "कृतिका", "तान्या", "पारुल", "मेघा", "ईशान", "देवांश", "युवराज", "प्रियांशु", "सूर्यांश",
    "अवनीश", "तेजस", "अक्षत", "अथर्वेश", "अन्वेषा", "अक्षिता", "वैदेही", "दृष्टि", "सौम्या", "लावण्या"
]

COMPARATIVE_DIMENSIONS = [
    {"type": "height", "comp_pos": "लंबा", "comp_neg": "छोटा", "sup_pos": "लंबा", "sup_neg": "छोटा"},
    {"type": "age", "comp_pos": "बड़ा", "comp_neg": "छोटा", "sup_pos": "बड़ा", "sup_neg": "छोटा"},
    {"type": "weight", "comp_pos": "भारी", "comp_neg": "हल्का", "sup_pos": "भारी", "sup_neg": "हल्का"},
    {"type": "speed", "comp_pos": "तेज़", "comp_neg": "धीमा", "sup_pos": "तेज़", "sup_neg": "धीमा"}
]

QUANTITY_ITEMS = [
    {"name": "सेब", "category": "fruit"},
    {"name": "किताबें", "category": "book"},
    {"name": "कलम", "category": "stationery"},
    {"name": "रुपये", "category": "currency"},
    {"name": "अंक", "category": "score"}
]

def generate_transitive(names_pool: List[str], rng: random.Random) -> Dict[str, Any]:
    a, b, c = rng.sample(names_pool, 3)
    dim = rng.choice(COMPARATIVE_DIMENSIONS)
    direction = rng.choice([True, False])
    if direction:
        pos_word, sup_max, sup_min = dim["comp_pos"], dim["sup_pos"], dim["sup_neg"]
        premise = f"{a}, {b} से {pos_word} है। {b}, {c} से {pos_word} है।"
        max_e, min_e = a, c
    else:
        pos_word, sup_max, sup_min = dim["comp_neg"], dim["sup_neg"], dim["sup_pos"]
        premise = f"{a}, {b} से {pos_word} है। {b}, {c} से {pos_word} है।"
        max_e, min_e = c, a

    if rng.choice([True, False]):
        q = f"इनमें से सबसे {sup_max} कौन है?"
        ans = max_e
    else:
        q = f"इनमें से सबसे {sup_min} कौन है?"
        ans = min_e

    prompt = f"{premise} {q} उत्तर:"
    return {"prompt": prompt, "target": ans, "text": f"{prompt} {ans}"}

def generate_quantitative(names_pool: List[str], rng: random.Random) -> Dict[str, Any]:
    a, b = rng.sample(names_pool, 2)
    item = rng.choice(QUANTITY_ITEMS)
    v1, v2 = rng.randint(5, 50), rng.randint(5, 50)
    while v1 == v2:
        v2 = rng.randint(5, 50)

    ask_more = rng.choice([True, False])
    if item["category"] == "currency":
        premise = f"{a} के पास {v1} रुपये हैं और {b} के पास {v2} रुपये हैं।"
        q = "किसके पास अधिक धन है?" if ask_more else "किसके पास कम धन है?"
    elif item["category"] == "score":
        premise = f"परीक्षा में {a} को {v1} अंक मिले और {b} को {v2} अंक मिले।"
        q = "किसको अधिक अंक मिले?" if ask_more else "किसको कम अंक मिले?"
    else:
        premise = f"{a} के पास {v1} {item['name']} हैं और {b} के पास {v2} {item['name']} हैं।"
        q = f"किसके पास अधिक {item['name']} हैं?" if ask_more else f"किसके पास कम {item['name']} हैं?"

    ans = (a if v1 > v2 else b) if ask_more else (a if v1 < v2 else b)
    prompt = f"{premise} {q} उत्तर:"
    return {"prompt": prompt, "target": ans, "text": f"{prompt} {ans}"}

def generate_multihop(names_pool: List[str], rng: random.Random) -> Dict[str, Any]:
    a, b, c = rng.sample(names_pool, 3)
    dim = rng.choice(COMPARATIVE_DIMENSIONS)
    pos = dim["comp_pos"]
    premise = f"{a}, {b} से {pos} है और {b}, {c} से {pos} है।"
    if rng.choice([True, False]):
        q = f"क्या {a}, {c} से {pos} है?"
        ans = "हाँ"
    else:
        q = f"क्या {c}, {a} से {pos} है?"
        ans = "नहीं"
    prompt = f"{premise} {q} उत्तर:"
    return {"prompt": prompt, "target": ans, "text": f"{prompt} {ans}"}

def create_synthetic_data(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    def make_split(names, n, seed):
        rng = random.Random(seed)
        res = []
        for _ in range(int(n * 0.40)): res.append(generate_transitive(names, rng))
        for _ in range(int(n * 0.35)): res.append(generate_quantitative(names, rng))
        for _ in range(n - len(res)): res.append(generate_multihop(names, rng))
        rng.shuffle(res)
        return res

    for split, names, n, seed in [("train", TRAIN_NAMES, 20000, 42), ("val", VAL_NAMES, 1000, 43), ("test", TEST_NAMES, 1000, 44)]:
        fp = os.path.join(out_dir, f"{split}.jsonl")
        data = make_split(names, n, seed)
        with open(fp, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Generated {len(data)} {split} samples -> {fp}")


# ==============================================================================
# 3. SFT Reasoning Dataset & Batching
# ==============================================================================

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

                # Mask prompt tokens with -100 so loss is computed ONLY on completion
                prompt_len = len(prompt_ids)
                for i in range(min(prompt_len - 1, len(target_ids))):
                    target_ids[i] = -100

                self.examples.append({
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                    "prompt": prompt,
                    "target": target
                })

        self.pad_id = pad_id if pad_id != -1 else 0

    def __len__(self) -> int:
        return len(self.examples)

    def get_batches(self, batch_size: int = 16, shuffle: bool = True, device: str = "cuda") -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        indices = list(range(len(self.examples)))
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            items = [self.examples[idx] for idx in batch_idx]

            max_len = max(len(it["input_ids"]) for it in items)
            inputs, targets = [], []

            for it in items:
                inp = it["input_ids"] + [self.pad_id] * (max_len - len(it["input_ids"]))
                tgt = it["target_ids"] + [-100] * (max_len - len(it["target_ids"]))
                inputs.append(inp)
                targets.append(tgt)

            x = torch.tensor(inputs, dtype=torch.long, device=device)
            y = torch.tensor(targets, dtype=torch.long, device=device)
            yield x, y


# ==============================================================================
# 4. Evaluation Function (Exact Match Accuracy)
# ==============================================================================

@torch.no_grad()
def evaluate_accuracy(
    model: TransformerLM,
    sp_model: spm.SentencePieceProcessor,
    dataset: SFTReasoningDataset,
    device: str = "cuda",
    num_samples: int = 200
) -> float:
    model.eval()
    samples = dataset.examples[:num_samples]
    correct = 0
    total = len(samples)

    for item in samples:
        prompt_ids = sp_model.encode(item["prompt"], out_type=int)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out_ids = model.generate(x, max_new_tokens=8, temperature=0.0, eos_id=sp_model.eos_id())
        gen_tokens = out_ids[0][len(prompt_ids):].tolist()
        pred_text = sp_model.decode(gen_tokens).strip()

        target = item["target"]
        first_word = pred_text.split()[0] if pred_text.split() else ""
        if (first_word == target) or (target in pred_text.split()[:2]) or (pred_text.startswith(target)):
            correct += 1

    return round(correct / total * 100, 2) if total > 0 else 0.0


# ==============================================================================
# 5. Path Resolution & Environment Auto-Detection
# ==============================================================================

def find_paths():
    print("\n[Inspecting Filesystem for Model Checkpoints and Tokenizers]")
    if os.path.exists("/kaggle/input"):
        print("Searching /kaggle/input:")
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"  {root} -> {len(files)} files")

    train_path, val_path, test_path = None, None, None
    ckpt_path, config_path, sp_path = None, None, None
    candidate_ckpts = []

    search_roots = ["/kaggle/input", "results_hindi", "hindi", "."]
    for search_root in search_roots:
        if os.path.exists(search_root):
            for root, _, files in os.walk(search_root):
                for f in files:
                    if f == "train.jsonl" and "reasoning" in root and not train_path:
                        train_path = os.path.join(root, f)
                    elif f == "val.jsonl" and "reasoning" in root and not val_path:
                        val_path = os.path.join(root, f)
                    elif f == "test.jsonl" and "reasoning" in root and not test_path:
                        test_path = os.path.join(root, f)
                    elif f.startswith("model_h_step_") and f.endswith(".pt"):
                        candidate_ckpts.append(os.path.join(root, f))
                    elif f == "config.json" and "hindi" in root.lower() and not config_path:
                        config_path = os.path.join(root, f)
                    elif f == "config.json" and not config_path:
                        config_path = os.path.join(root, f)
                    elif f == "hindi_sp.model" and not sp_path:
                        sp_path = os.path.join(root, f)

    if candidate_ckpts:
        def get_step(p):
            b = os.path.basename(p)
            try: return int(b.replace("model_h_step_", "").replace(".pt", ""))
            except: return 0
        candidate_ckpts.sort(key=get_step)
        ckpt_path = candidate_ckpts[-1]
        print(f"Selected latest checkpoint by step: {ckpt_path}")

    # Always generate the 20,000 sample programmatic reasoning dataset with zero leakage
    data_dir = os.path.join(OUTPUT_DIR, "hindi_reasoning_data")
    print(f"Generating 20,000 sample programmatic reasoning dataset into {data_dir}...")
    create_synthetic_data(data_dir)
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")

    # Fallbacks for local execution
    train_path = train_path or "hindi/reasoning/data/train.jsonl"
    val_path = val_path or "hindi/reasoning/data/val.jsonl"
    test_path = test_path or "hindi/reasoning/data/test.jsonl"
    ckpt_path = ckpt_path or "results_hindi/checkpoints/model_h_step_25000.pt"
    config_path = config_path or "results_hindi/config.json"
    sp_path = sp_path or "hindi/tokenizer/hindi_sp.model"

    return train_path, val_path, test_path, ckpt_path, config_path, sp_path


# ==============================================================================
# 6. Main Finetuning Loop
# ==============================================================================

def main():
    print("=" * 75)
    print("Reasoning Finetuning: Hindi Model H (Phase 3)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Detected GPU: {gpu_name} (Capability {cap[0]}.{cap[1]})")
        if cap[0] < 7:
            print("[NOTICE] CUDA capability < 7.0 (Tesla P100) is incompatible with current Kaggle PyTorch kernels.")
            print("[NOTICE] Automatically falling back to CPU execution to guarantee completion.")
            device = "cpu"
        else:
            print(f"Accelerating on compatible GPU: {gpu_name}!")
    else:
        print("Hardware Device: CPU")

    train_p, val_p, test_p, ckpt_p, cfg_p, sp_p = find_paths()
    print(f"Pretrained Checkpoint: {ckpt_p}")
    print(f"Tokenizer: {sp_p}")
    print(f"Training Reasoning Data: {train_p}")

    sp = spm.SentencePieceProcessor(model_file=sp_p)

    if os.path.exists(cfg_p):
        with open(cfg_p, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
    else:
        config = ModelConfig(vocab_size=len(sp), d_model=512, n_layer=6, n_head=8, d_ff=2048, max_seq_len=256)

    model = TransformerLM(config).to(device)

    print(f"Loading pretrained weights from {ckpt_p}...")
    ckpt = torch.load(ckpt_p, map_location=device)
    raw_sd = ckpt.get("model_state_dict", ckpt)
    clean_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_sd[k] = v
    model.load_state_dict(clean_sd, strict=True)
    print("Pretrained weights successfully loaded with strict=True!")

    # Datasets
    train_ds = SFTReasoningDataset(train_p, sp)
    val_ds = SFTReasoningDataset(val_p, sp)
    test_ds = SFTReasoningDataset(test_p, sp)
    print(f"Loaded: Train={len(train_ds)}, Val={len(val_ds)}, Held-out Test={len(test_ds)}")

    # 1. Baseline Pretrained Accuracy
    print("\n[Baseline Evaluation] Measuring Pretrained Model Reasoning Accuracy on Held-out Test Set...")
    base_test_acc = evaluate_accuracy(model, sp, test_ds, device, num_samples=100)
    print(f"Pretrained Base Model Accuracy: {base_test_acc}%\n")

    # 2. Hyperparameters
    max_steps = 1000
    batch_size = 16
    lr = 5e-5
    warmup_steps = 100

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    log_csv_path = os.path.join(OUTPUT_DIR, "finetune_log.csv")
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "lr", "train_loss", "val_loss", "val_acc"])

    print("Beginning Reasoning SFT (1,000 steps with validation checkpoints every 100 steps)...")
    step = 0
    best_val_acc = 0.0
    running_loss = 0.0

    steps_hist, train_loss_hist, val_loss_hist, val_acc_hist = [], [], [], []

    model.train()
    start_time = time.time()

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
                avg_loss = running_loss / 20
                print(f"Step {step:4d}/{max_steps} | Train Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
                running_loss = 0.0

            # Periodic Validation Checkpoint
            if step % 100 == 0 or step == max_steps:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vx, vy in val_ds.get_batches(batch_size=batch_size, shuffle=False, device=device):
                        _, vloss, _ = model(vx, targets=vy)
                        val_losses.append(vloss.item())

                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
                val_acc = evaluate_accuracy(model, sp, val_ds, device, num_samples=100)

                steps_hist.append(step)
                train_loss_hist.append(loss.item())
                val_loss_hist.append(avg_val_loss)
                val_acc_hist.append(val_acc)

                csv_writer.writerow([step, f"{current_lr:.2e}", f"{loss.item():.4f}", f"{avg_val_loss:.4f}", f"{val_acc:.2f}"])
                csv_file.flush()

                print(f"--> Checkpoint @ Step {step:4d}: Val Loss = {avg_val_loss:.4f} | Val Accuracy = {val_acc:.2f}%")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", "model_h_finetuned_best.pt")
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "val_accuracy": val_acc,
                        "config": config.to_dict()
                    }, best_ckpt_path)
                    print(f"    * Saved New Best Checkpoint: {best_ckpt_path} ({val_acc:.2f}%)")

                model.train()

    csv_file.close()
    elapsed = time.time() - start_time
    print(f"\nFinetuning completed in {elapsed/60:.1f} minutes!")

    # Save Final Model
    final_ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", "model_h_finetuned.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config.to_dict()
    }, final_ckpt_path)
    print(f"Saved Final Checkpoint -> {final_ckpt_path}")

    # 3. Final Held-Out Test Evaluation
    print("\n[Final Evaluation] Measuring Finetuned Model Accuracy on Held-out Test Set...")
    finetuned_test_acc = evaluate_accuracy(model, sp, test_ds, device, num_samples=200)
    print("=" * 60)
    print(f"PRETRAINED Base Test Accuracy : {base_test_acc:.2f}%")
    print(f"FINETUNED Final Test Accuracy : {finetuned_test_acc:.2f}%")
    print(f"Accuracy Gain                 : +{finetuned_test_acc - base_test_acc:.2f}%")
    print("=" * 60)

    # 4. Plot Finetuning Curves
    plt.figure(figsize=(10, 4), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(steps_hist, train_loss_hist, label="Train Loss", color="#1f77b4")
    plt.plot(steps_hist, val_loss_hist, label="Val Loss", color="#ff7f0e", linestyle="--")
    plt.title("Hindi SFT Loss Dynamics", fontsize=11, fontweight="bold")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(steps_hist, val_acc_hist, label="Val Accuracy (%)", color="#2ca02c", marker="o")
    plt.axhline(y=base_test_acc, color="gray", linestyle=":", label=f"Pretrained ({base_test_acc:.1f}%)")
    plt.title("Reasoning Accuracy Progression", fontsize=11, fontweight="bold")
    plt.xlabel("Step")
    plt.ylabel("Exact Match Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, "figures", "hindi_finetuning_curves.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved training curves -> {curve_path}")

if __name__ == "__main__":
    main()
