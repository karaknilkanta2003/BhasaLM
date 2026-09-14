"""
Comprehensive Phase 2 Evaluation and Attention Analysis Suite
============================================================
Computes:
1. Perplexity (PPL) and Bits-Per-Byte (BPB) on held-out test splits.
2. Text generation across decoding strategies: Greedy, Temp 0.5, Temp 1.0, Temp 1.5.
3. Automated reference metrics: BLEU-4, chrF++, ROUGE-L, Distinct-1/2, repetition rate.
4. Internal Attention Analysis: Entropy, Mean Attention Distance, Head Specialization.
5. High-resolution attention heatmaps comparing Layer 1 vs. Layer 6.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import math
import torch
import sentencepiece as spm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure font support for Indic scripts (Devanagari and Assamese/Bengali)
plt.rcParams['font.sans-serif'] = ['Nirmala UI', 'Segoe UI', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from common.model import TransformerLM, ModelConfig
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

def load_model(checkpoint_path, config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)
    model = TransformerLM(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config

def evaluate_lm(model, sp, test_path, num_batches=100, batch_size=32, seq_len=256):
    """Calculates test loss, PPL, and BPB on held-out test split."""
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
    tok_count = 50000

    bytes_per_token = total_bytes / tok_count if tok_count > 0 else 1.0
    bpb = avg_loss / (math.log(2.0) * bytes_per_token)

    return {
        "test_loss": round(avg_loss, 4),
        "test_ppl": round(ppl, 2),
        "test_bpb": round(bpb, 4),
        "bytes_per_token": round(bytes_per_token, 3)
    }

def run_generation_experiments(model, sp, prompts, references):
    """Evaluates greedy vs. temperature sampling."""
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
                {"prompt": p, "continuation": g[len(p):].strip()}
                for p, g in zip(prompts[:2], gen_outputs[:2])
            ]
        }

    return results

def plot_attention_heatmaps(model, sp, sentence, lang_name, save_path):
    """Plots Layer 1 vs Layer 6 attention heatmaps for a representative sentence."""
    tokens = sp.encode_as_pieces(sentence)
    token_ids = sp.encode_as_ids(sentence)
    
    # Cap tokens for legible heatmap
    tokens = tokens[:16]
    token_ids = token_ids[:16]
    clean_tokens = [t.replace("\u2581", "_") for t in tokens]

    inp = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, _, attentions = model(inp, return_attentions=True)

    # Layer 1 and Layer 6
    l1_attn = attentions[0][0].cpu().numpy() # (n_head, T, T)
    l6_attn = attentions[5][0].cpu().numpy() # (n_head, T, T)

    # Pick Head 0 for visualization
    l1_h0 = l1_attn[0]
    l6_h0 = l6_attn[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Plot Layer 1 Head 0 (Early layer: Positional / Local)
    sns.heatmap(
        l1_h0,
        xticklabels=clean_tokens,
        yticklabels=clean_tokens,
        cmap="Blues",
        ax=axes[0],
        cbar_kws={"label": "Attention Weight"}
    )
    axes[0].set_title(f"{lang_name} Layer 1 (Head 1): Local / Diagonal Alignment", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Key Position (Attended)", fontsize=10)
    axes[0].set_ylabel("Query Position (Current)", fontsize=10)

    # Plot Layer 6 Head 0 (Late layer: Semantic / Diffuse)
    sns.heatmap(
        l6_h0,
        xticklabels=clean_tokens,
        yticklabels=clean_tokens,
        cmap="Purples",
        ax=axes[1],
        cbar_kws={"label": "Attention Weight"}
    )
    axes[1].set_title(f"{lang_name} Layer 6 (Head 1): Semantic / Contextual Dispersion", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Key Position (Attended)", fontsize=10)
    axes[1].set_ylabel("Query Position (Current)", fontsize=10)

    plt.suptitle(f"Self-Attention Dynamics: Early (Layer 1) vs. Late (Layer 6) - {lang_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved attention heatmap: {save_path}")

    # Compute layer-level stats
    stats = analyze_layer_attentions(attentions)
    l1_entropy = [s["entropy"] for s in stats if s["layer"] == 1]
    l6_entropy = [s["entropy"] for s in stats if s["layer"] == 6]
    l1_dist = [s["mean_distance"] for s in stats if s["layer"] == 1]
    l6_dist = [s["mean_distance"] for s in stats if s["layer"] == 6]

    return {
        "l1_avg_entropy": round(sum(l1_entropy) / len(l1_entropy), 3),
        "l6_avg_entropy": round(sum(l6_entropy) / len(l6_entropy), 3),
        "l1_avg_distance": round(sum(l1_dist) / len(l1_dist), 2),
        "l6_avg_distance": round(sum(l6_dist) / len(l6_dist), 2),
        "head_summary": stats
    }

def main():
    print("=" * 70)
    print("Running Full Phase 2 Evaluation and Attention Analysis")
    print("=" * 70)

    os.makedirs("report/phase2/figures", exist_ok=True)

    # 1. Models and Tokenizers
    sp_h = spm.SentencePieceProcessor(model_file="hindi/tokenizer/hindi_sp.model")
    model_h, _ = load_model("results_hindi/checkpoints/model_h_step_25000.pt", "results_hindi/config.json")
    print(f"Loaded Model H (Hindi) with vocab {len(sp_h)}")

    sp_l = spm.SentencePieceProcessor(model_file="assamese/tokenizer/assamese_sp.model")
    model_l, _ = load_model("results_assamese/checkpoints/model_l_step_25000.pt", "results_assamese/config.json")
    print(f"Loaded Model L (Assamese) with vocab {len(sp_l)}")

    # 2. Perplexity & BPB
    print("\n[1/3] Computing Held-out Test PPL & BPB...")
    lm_h = evaluate_lm(model_h, sp_h, "hindi/data/test.jsonl")
    print("Model H (Hindi) Test Metrics:", lm_h)

    lm_l = evaluate_lm(model_l, sp_l, "assamese/data/test.jsonl")
    print("Model L (Assamese) Test Metrics:", lm_l)

    # 3. Generation Experiments
    print("\n[2/3] Running Generation Experiments across Temperatures...")
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
    gen_h = run_generation_experiments(model_h, sp_h, hindi_prompts, hindi_refs)

    assamese_prompts = [
        "ভাৰত এখন বিশাল আৰু বৈচিত্ৰ্যময় দেশ য'ত",
        "বিজ্ঞান আৰু প্ৰযুক্তিবিদ্যাৰ উন্নয়নৰ বাবে",
        "শিক্ষা ব্যৱস্থাত নতুন পৰিৱৰ্তন আনিবলৈ",
        "প্ৰাকৃতিক সৌন্দৰ্য্যৰে ভৰপূৰ অসমৰ",
        "আধুনিক সমাজত যোগাযোগ ব্যৱস্থাৰ গুৰুত্ব"
    ]
    assamese_refs = [
        "ভাৰত এখন বিশাল আৰু বৈচিত্ৰ্যময় দেশ য'ত বিভিন্ন জাতি-জনগোষ্ঠীৰ লোকে বাস কৰে।",
        "বিজ্ঞান আৰু প্ৰযুক্তিবিদ্যাৰ উন্নয়নৰ বাবে নতুন নতুন গৱেষণা অতি প্ৰয়োজনীয়।",
        "শিক্ষা ব্যৱস্থাত নতুন পৰিৱৰ্তন আনিবলৈ ছাত্ৰ-ছাত্ৰীসকলক উৎসাহিত কৰা হৈছে।",
        "প্ৰাকৃতিক সৌন্দৰ্য্যৰে ভৰপূৰ অসমৰ চাহ বাগিচা আৰু নদীসমূহে সকলোকে মুগ্ধ কৰে।",
        "আধুনিক সমাজত যোগাযোগ ব্যৱস্থাৰ গুৰুত্ব দিনক দিনে বৃদ্ধি পাবলৈ ধৰিছে।"
    ]
    gen_l = run_generation_experiments(model_l, sp_l, assamese_prompts, assamese_refs)

    # 4. Attention Analysis & Heatmaps
    print("\n[3/3] Generating Attention Visualizations & Entropy Metrics...")
    attn_h = plot_attention_heatmaps(
        model_h, sp_h,
        "भारत एक महान देश है जहाँ विभिन्न संस्कृतियों का संगम है।",
        "Hindi (Model H)",
        "report/phase2/figures/hindi_attention_layer1_vs_6.png"
    )

    attn_l = plot_attention_heatmaps(
        model_l, sp_l,
        "ভাৰত এখন বিশাল দেশ য'ত বিভিন্ন জাতিৰ লোকে বাস কৰে।",
        "Assamese (Model L)",
        "report/phase2/figures/assamese_attention_layer1_vs_6.png"
    )

    # Compile into JSON
    full_results = {
        "language_modeling": {
            "hindi_model_h": lm_h,
            "assamese_model_l": lm_l
        },
        "generation": {
            "hindi_model_h": gen_h,
            "assamese_model_l": gen_l
        },
        "attention_analysis": {
            "hindi_model_h": attn_h,
            "assamese_model_l": attn_l
        }
    }

    out_json = "report/phase2/eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll Phase 2 evaluation results saved to: {out_json}")

if __name__ == "__main__":
    main()
