import json
from pathlib import Path

results_dir = Path(r"d:\mri\results")

print("=== TRAINING PROGRESS ===")
for f in sorted(results_dir.glob("*_history.json")):
    data = json.load(open(f))
    name = data.get("model_name", f.stem)
    acc = data.get("best_val_acc", 0)
    epochs = data.get("epochs_trained", 0)
    time_s = data.get("training_time_seconds", 0)
    print(f"  {name}: {acc:.2f}% val acc | {epochs} epochs | {time_s/60:.1f} min")

models_dir = Path(r"d:\mri\models")
print("\n=== SAVED MODEL FILES ===")
for f in sorted(models_dir.glob("*.pt")):
    print(f"  {f.name} ({f.stat().st_size/1024/1024:.1f} MB)")

summary = results_dir / "training_summary.json"
if summary.exists():
    data = json.load(open(summary))
    print("\n=== TRAINING SUMMARY ===")
    for name, info in data.get("models", {}).items():
        print(f"  {name}: {info}")
