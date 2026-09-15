import csv
import matplotlib.pyplot as plt
import numpy as np

# Configure publication aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def read_log(path):
    steps, train_loss, val_loss, val_acc = [], [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            val_acc.append(float(row['val_acc']))
    return steps, train_loss, val_loss, val_acc

hi_steps, hi_train, hi_val, hi_acc = read_log('results_hindi_reasoning/finetune_log.csv')
as_steps, as_train, as_val, as_acc = read_log('results_assamese_reasoning/finetune_log.csv')

# -------------------------------------------------------------
# 1. Combined Finetuning Loss Curves
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# Loss Comparison
ax1.plot(hi_steps, hi_train, color='#1f77b4', linewidth=2, label='Model H (Hindi) Train Loss')
ax1.plot(hi_steps, hi_val, color='#1f77b4', linestyle='--', linewidth=2, label='Model H (Hindi) Val Loss')
ax1.plot(as_steps, as_train, color='#ff7f0e', linewidth=2, label='Model L (Assamese) Train Loss')
ax1.plot(as_steps, as_val, color='#ff7f0e', linestyle='--', linewidth=2, label='Model L (Assamese) Val Loss')
ax1.set_title('Cross-Entropy Loss Dynamics (20K SFT, Kaggle GPU)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Training Step', fontsize=11)
ax1.set_ylabel('Masked Cross-Entropy Loss', fontsize=11)
ax1.legend(loc='upper right', frameon=True)
ax1.grid(True, alpha=0.3)

# Accuracy Comparison
ax2.plot(hi_steps, hi_acc, color='#1f77b4', marker='o', linewidth=2, label='Model H (Hindi) Val Acc (Peak: 65%)')
ax2.plot(as_steps, as_acc, color='#ff7f0e', marker='s', linewidth=2, label='Model L (Assamese) Val Acc (Peak: 44%)')
ax2.axhline(y=2.0, color='#1f77b4', linestyle=':', alpha=0.6, label='Model H Pretrained Base (2.0%)')
ax2.axhline(y=0.0, color='#ff7f0e', linestyle=':', alpha=0.6, label='Model L Pretrained Base (0.0%)')
ax2.set_title('Validation Exact Match Accuracy (%) Across Steps', fontsize=12, fontweight='bold')
ax2.set_xlabel('Training Step', fontsize=11)
ax2.set_ylabel('Exact Match Accuracy (%)', fontsize=11)
ax2.set_ylim(-2, 75)
ax2.legend(loc='upper left', frameon=True)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report/phase3/figures/finetuning_loss_curves.png', dpi=300)
plt.close()
print("Saved report/phase3/figures/finetuning_loss_curves.png")

# -------------------------------------------------------------
# 2. Reasoning Accuracy Comparison Bar Chart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

categories = ['Base Pretrained', 'Val Checkpoint Peak', 'Final Held-Out Test']
hindi_vals = [2.0, 65.0, 60.0]
assamese_vals = [0.0, 44.0, 45.0]

x = np.arange(len(categories))
width = 0.35

rects1 = ax.bar(x - width/2, hindi_vals, width, label='Model H (Hindi — High Resource)', color='#1f77b4', alpha=0.9, edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x + width/2, assamese_vals, width, label='Model L (Assamese — Low Resource)', color='#ff7f0e', alpha=0.9, edgecolor='black', linewidth=0.5)

ax.set_title('Reasoning Accuracy: Pretrained vs. 20K SFT Finetuned', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Exact Match Accuracy (%)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='semibold')
ax.set_ylim(0, 75)
ax.legend(loc='upper left', frameon=True, fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels on top of bars
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1, ax)
autolabel(rects2, ax)

plt.tight_layout()
plt.savefig('report/phase3/figures/reasoning_accuracy_comparison.png', dpi=300)
plt.close()
print("Saved report/phase3/figures/reasoning_accuracy_comparison.png")
