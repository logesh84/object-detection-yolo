import os
import sys
import random
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] Run: pip install ultralytics")


# CONFIG
DATASET_YAML = "dataset/data.yaml"
DATASET_DIR  = "dataset"
BASE_MODEL   = "yolov8s.pt"
EPOCHS       = 100
IMAGE_SIZE   = 640
BATCH_SIZE   = 16
DEVICE       = "cpu"

PROJECT  = "runs/detect"
RUN_NAME = "screw_nut_bolt"


def check_dataset():
    if not os.path.exists(DATASET_YAML):
        sys.exit(f"[ERROR] Missing {DATASET_YAML}")
    print(f"[✓] Dataset found : {DATASET_YAML}")


def fix_dataset_structure():
    train_images = Path(DATASET_DIR) / "train/images"
    train_labels = Path(DATASET_DIR) / "train/labels"

    valid_images = Path(DATASET_DIR) / "valid/images"
    valid_labels = Path(DATASET_DIR) / "valid/labels"

    # If valid folder already exists → skip
    if valid_images.exists():
        print("[✓] Valid folder already exists")
        return

    print("[⚠] 'valid' folder missing → creating automatically...")

    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)

    images = list(train_images.glob("*.*"))
    random.shuffle(images)

    split_count = int(len(images) * 0.2)  # 20% for validation

    for img_path in images[:split_count]:
        label_path = train_labels / (img_path.stem + ".txt")

        shutil.move(str(img_path), valid_images / img_path.name)

        if label_path.exists():
            shutil.move(str(label_path), valid_labels / label_path.name)

    print(f"[✓] Moved {split_count} images to validation set")


def train():
    check_dataset()
    fix_dataset_structure()

    print(f"\n{'='*55}")
    print(f"  Starting YOLO Training (screw, nut, bolt)")
    print(f"  Base model  : {BASE_MODEL}")
    print(f"  Dataset     : {DATASET_YAML}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Image size  : {IMAGE_SIZE}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Device      : {'GPU' if DEVICE == 0 else 'CPU'}")
    print(f"{'='*55}\n")

    model = YOLO(BASE_MODEL)

    model.train(
        data     = DATASET_YAML,
        epochs   = EPOCHS,
        imgsz    = IMAGE_SIZE,
        batch    = BATCH_SIZE,
        device   = DEVICE,
        project  = PROJECT,
        name     = RUN_NAME,
        patience = 20,
        seed     = 42,
        verbose  = True,
    )

    best_model = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"

    print(f"\n{'='*55}")
    print(f"  ✅ Training Complete!")
    print(f"  Best model → {best_model}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    train()