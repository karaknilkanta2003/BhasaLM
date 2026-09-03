"""
Loss Curve Plotting Suite for Phase 2 Pretraining
=================================================
Plots publication-quality comparative loss curves and validation perplexity curves
for Hindi Model H vs. Assamese Model L.
"""

import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def load_log_data(csv_path):
    steps = []
    train_losses = []
    val_losses = []
    val_ppls = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            val_ppls.append(float(row["val_ppl"]))
    return steps, train_losses, val_losses, val_ppls

def plot_curves():
    os.makedirs("report/phase2/figures", exist_ok=True)
    
    hindi_csv = "results_hindi/training_log.csv"
    as_csv = "results_assamese/training_log.csv"

    h_steps, h_tr_loss, h_val_loss, h_val_ppl = load_log_data(hindi_csv)
    as_steps, as_tr_loss, as_val_loss, as_val_ppl = load_log_data(as_csv)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # -------------------------------------------------------------
    # Figure 1: Cross-Entropy Loss Curves Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Hindi
    ax.plot(h_steps, h_tr_loss, color="#1f77b4", linestyle="--", alpha=0.7, label="Model H (Hindi) - Train Loss")
    ax.plot(h_steps, h_val_loss, color="#1f77b4", linewidth=2.5, label="Model H (Hindi) - Val Loss")
    
    # Assamese
    ax.plot(as_steps, as_tr_loss, color="#ff7f0e", linestyle="--", alpha=0.7, label="Model L (Assamese) - Train Loss")
    ax.plot(as_steps, as_val_loss, color="#ff7f0e", linewidth=2.5, label="Model L (Assamese) - Val Loss")

    ax.set_title("Pretraining Loss Trajectory: Model H (Hindi) vs. Model L (Assamese)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Training Steps (Batch Size: 64, Context: 256 tokens)", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_xlim(0, 25000)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)
    plt.tight_layout()
    
    loss_fig_path = "report/phase2/figures/loss_curves.png"
    plt.savefig(loss_fig_path)
    plt.close()
    print(f"Saved: {loss_fig_path}")

    # -------------------------------------------------------------
    # Figure 2: Validation Perplexity Curves (Log Scale)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax.plot(h_steps, h_val_ppl, color="#1f77b4", linewidth=2.5, marker="o", markersize=4, label="Model H (Hindi) - Val PPL")
    ax.plot(as_steps, as_val_ppl, color="#ff7f0e", linewidth=2.5, marker="s", markersize=4, label="Model L (Assamese) - Val PPL")

    ax.set_yscale("log")
    ax.set_title("Held-out Validation Perplexity (Log Scale): Model H vs. Model L", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Validation Perplexity (exp(Loss))", fontsize=12)
    ax.set_xlim(0, 25000)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)
    plt.tight_layout()
    
    ppl_fig_path = "report/phase2/figures/perplexity_curves.png"
    plt.savefig(ppl_fig_path)
    plt.close()
    print(f"Saved: {ppl_fig_path}")

if __name__ == "__main__":
    plot_curves()
