import csv

def analyze_log(path, lang):
    print("=" * 65)
    print(f"ANALYSIS: {lang}")
    print("=" * 65)
    with open(path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    print(f"Total Periodic Evaluations Logged: {len(reader)}")
    if not reader:
        print("Empty log!")
        return
    first = reader[0]
    last = reader[-1]
    print(f"Initial Step {first['step']}: Train Loss = {float(first['train_loss']):.4f} | Val Loss = {float(first['val_loss']):.4f} | Val PPL = {float(first['val_ppl']):.2f}")
    print(f"Final Step   {last['step']}: Train Loss = {float(last['train_loss']):.4f} | Val Loss = {float(last['val_loss']):.4f} | Val PPL = {float(last['val_ppl']):.2f}")
    tokens = int(last["tokens_seen"])
    print(f"Total Tokens Processed: {tokens:,} ({tokens / 1e6:.1f} Million Tokens)")
    
    indices = [0, len(reader) // 4, len(reader) // 2, 3 * len(reader) // 4, len(reader) - 1]
    print("\nConvergence Trajectory:")
    for idx in indices:
        r = reader[idx]
        step = r["step"]
        tl = float(r["train_loss"])
        vl = float(r["val_loss"])
        ppl = float(r["val_ppl"])
        spd = r["tokens_per_sec"]
        print(f"  Step {step:>5}: Train Loss = {tl:.4f} | Val Loss = {vl:.4f} | Val PPL = {ppl:7.2f} | Speed = {spd} tok/s")
    print()

if __name__ == "__main__":
    analyze_log("results_hindi/training_log.csv", "HINDI MODEL H (~25M)")
    analyze_log("results_assamese/training_log.csv", "ASSAMESE MODEL L (~25M)")
