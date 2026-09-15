import sys, os, time, math, torch, json, csv, random
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())
from common.model import TransformerLM, ModelConfig
import sentencepiece as spm
import matplotlib.pyplot as plt

def run_finetuning(lang, base_ckpt, sp_file, train_file, val_file, test_file, out_dir, save_name, max_steps=600):
    print(f"\n{'='*70}\n[START] 20K SFT Finetuning for {lang.upper()} ({max_steps} steps)\n{'='*70}")
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs("report/phase3/figures", exist_ok=True)
    
    sp = spm.SentencePieceProcessor(model_file=sp_file)
    config = ModelConfig(vocab_size=len(sp), d_model=512, n_layer=6, n_head=8, d_ff=2048, max_seq_len=256)
    model = TransformerLM(config)
    
    # Load pretrained weights with strict=True
    base_sd = torch.load(base_ckpt, map_location='cpu', weights_only=False)['model_state_dict']
    clean_sd = {k.replace("module.", ""): v for k, v in base_sd.items()}
    model.load_state_dict(clean_sd, strict=True)
    print(f"Pretrained base model loaded from {base_ckpt} with strict=True!")

    # Build dataset
    def load_ds(path):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                prompt_ids = sp.encode(d['prompt'], out_type=int)
                full_ids = sp.encode(d['text'], out_type=int) + [sp.eos_id()]
                input_ids = full_ids[:-1]
                target_ids = list(full_ids[1:])
                # Mask prompt tokens
                for i in range(min(len(prompt_ids) - 1, len(target_ids))):
                    target_ids[i] = -100
                examples.append({
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                    "prompt": d['prompt'],
                    "target": d['target'].strip(),
                    "task_type": d.get('task_type', '')
                })
        return examples

    train_data = load_ds(train_file)
    val_data = load_ds(val_file)
    print(f"Loaded: Train={len(train_data):,} samples, Val={len(val_data):,} samples")

    def eval_metrics(max_eval=60):
        model.eval()
        correct = 0
        eos_id = sp.eos_id()
        sub = val_data[:max_eval]
        
        # Calculate val loss
        v_losses = []
        with torch.no_grad():
            for i in range(0, len(sub), 15):
                b_items = sub[i:i+15]
                max_l = max(len(it["input_ids"]) for it in b_items)
                pad_x = [it["input_ids"] + [0]*(max_l-len(it["input_ids"])) for it in b_items]
                pad_y = [it["target_ids"] + [-100]*(max_l-len(it["target_ids"])) for it in b_items]
                vx = torch.tensor(pad_x, dtype=torch.long)
                vy = torch.tensor(pad_y, dtype=torch.long)
                _, vloss, _ = model(vx, targets=vy)
                v_losses.append(vloss.item())

            for it in sub:
                p_ids = sp.encode(it['prompt'], out_type=int)
                out = model.generate(torch.tensor([p_ids]), max_new_tokens=8, temperature=0.0, eos_id=eos_id)
                pred = sp.decode(out[0, len(p_ids):].tolist()).strip()
                first = pred.split()[0] if pred.split() else ""
                tgt = it['target']
                if first == tgt or tgt in pred or pred.startswith(tgt):
                    correct += 1

        avg_vloss = sum(v_losses) / len(v_losses) if v_losses else 0.0
        acc = round(correct / len(sub) * 100, 2)
        return avg_vloss, acc

    opt = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    batch_size = 16
    step = 0
    best_acc = 0.0
    running_loss = 0.0

    log_path = os.path.join(out_dir, "finetune_log.csv")
    csv_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "train_loss", "val_loss", "val_acc"])

    steps_hist, train_hist, val_hist, acc_hist = [], [], [], []

    model.train()
    t0 = time.time()
    indices = list(range(len(train_data)))
    
    while step < max_steps:
        random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            if step >= max_steps: break
            b_idx = indices[i:i+batch_size]
            b_in = [train_data[idx]["input_ids"] for idx in b_idx]
            b_tgt = [train_data[idx]["target_ids"] for idx in b_idx]
            max_len = max(len(b) for b in b_in)
            pad_in = [b + [0]*(max_len-len(b)) for b in b_in]
            pad_tgt = [b + [-100]*(max_len-len(b)) for b in b_tgt]
            
            x = torch.tensor(pad_in, dtype=torch.long)
            y = torch.tensor(pad_tgt, dtype=torch.long)
            
            opt.zero_grad()
            _, loss, _ = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            running_loss += loss.item()
            step += 1

            if step % 50 == 0 or step == max_steps:
                vloss, val_acc = eval_metrics(60)
                avg_l = running_loss / 50
                running_loss = 0.0
                print(f"Step {step:4d}/{max_steps} | Train Loss: {avg_l:.4f} | Val Loss: {vloss:.4f} | Val Acc: {val_acc:.1f}% | Elapsed: {time.time()-t0:.1f}s")
                writer.writerow([step, f"{avg_l:.4f}", f"{vloss:.4f}", f"{val_acc:.1f}"])
                csv_file.flush()

                steps_hist.append(step)
                train_hist.append(avg_l)
                val_hist.append(vloss)
                acc_hist.append(val_acc)
                
                if val_acc >= best_acc:
                    best_acc = val_acc
                    best_path = os.path.join(out_dir, "checkpoints", f"{save_name}_best.pt")
                    torch.save({"model_state_dict": model.state_dict(), "step": step, "val_accuracy": val_acc}, best_path)
                model.train()

    final_path = os.path.join(out_dir, "checkpoints", f"{save_name}.pt")
    torch.save({"model_state_dict": model.state_dict(), "step": step, "val_accuracy": best_acc}, final_path)
    csv_file.close()

    # Generate loss curve figure
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_title(f"{lang.capitalize()} 20K SFT Loss Dynamics", fontsize=13, fontweight="bold")
    ax1.plot(steps_hist, train_hist, 'b-', label="Train Loss", linewidth=2)
    ax1.plot(steps_hist, val_hist, 'r--', label="Val Loss", linewidth=2)
    ax1.set_xlabel("Steps", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(steps_hist, acc_hist, 'g-', label="Val Accuracy (%)", linewidth=2)
    ax2.set_ylabel("Val Accuracy (%)", color='g', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='g')

    plt.tight_layout()
    plot_p = os.path.join(out_dir, "figures", f"{lang}_finetuning_curves.png")
    plt.savefig(plot_p, dpi=150)
    plt.savefig(f"report/phase3/figures/{lang}_finetuning_curves.png", dpi=150)
    plt.close()

    print(f"[COMPLETE] {lang.upper()} Finetuned. Best Val Acc: {best_acc}%. Saved to {final_path}")

if __name__ == "__main__":
    run_finetuning(
        'hindi',
        'results_hindi/checkpoints/model_h_step_25000.pt',
        'hindi/tokenizer/hindi_sp.model',
        'hindi/reasoning/data/train.jsonl',
        'hindi/reasoning/data/val.jsonl',
        'hindi/reasoning/data/test.jsonl',
        'results_hindi_reasoning',
        'model_h_finetuned',
        max_steps=600
    )
    run_finetuning(
        'assamese',
        'results_assamese/checkpoints/model_l_step_25000.pt',
        'assamese/tokenizer/assamese_sp.model',
        'assamese/reasoning/data/train.jsonl',
        'assamese/reasoning/data/val.jsonl',
        'assamese/reasoning/data/test.jsonl',
        'results_assamese_reasoning',
        'model_l_finetuned',
        max_steps=600
    )

