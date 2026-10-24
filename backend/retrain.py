import shutil
import sys
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent
BEST = BACKEND / "best.pt"
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30

# Original dataset ke class names lo
orig = yaml.safe_load((ROOT / "dataset" / "data.yaml").read_text())
names = orig["names"]

# purana train data + naya data, validation purane valid set par
cfg = {
    "path": str(ROOT),
    "train": ["dataset/train/images", "storage/new_data/images"],
    "val": "dataset/valid/images",
    "names": names,
}
cfg_path = ROOT / "storage" / "retrain_data.yaml"
cfg_path.write_text(yaml.dump(cfg))

print("Purane model ka mAP nikal rahe hain...")
old_map = YOLO(str(BEST)).val(data=str(cfg_path), verbose=False).box.map
print(f"Old mAP50-95: {old_map:.4f}")

run_name = f"retrain_{int(time.time())}"
print("Training start...")
model = YOLO(str(BEST))                       # purane weights se fine-tune
model.train(data=str(cfg_path), epochs=epochs, imgsz=640, batch=8,
            project=str(ROOT / "runs"), name=run_name)

new_best = ROOT / "runs" / run_name / "weights" / "best.pt"
new_map = YOLO(str(new_best)).val(data=str(cfg_path), verbose=False).box.map
print(f"New mAP50-95: {new_map:.4f}")

if new_map > old_map:
    backup = BACKEND / f"best_backup_{int(time.time())}.pt"
    shutil.copy(BEST, backup)
    shutil.copy(new_best, BEST)
    print(f"RESULT: naya model behtar hai, replace ho gaya. Backup: {backup.name}")
else:
    print("RESULT: naya model behtar nahi tha, purana model rakha gaya.")