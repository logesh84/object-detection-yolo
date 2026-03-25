"""
╔══════════════════════════════════════════════════════════════╗
║              YOLO TRAINER  —  train.py                      ║
║   Trains YOLOv8 on your screw & nut dataset                 ║
╚══════════════════════════════════════════════════════════════╝

BEFORE RUNNING:
---------------
  1. Capture images using capture.py
  2. Upload to Roboflow → annotate → export as YOLOv8 format
  3. Extract the zip into this folder
  4. Update DATASET_YAML below to point to your data.yaml

HOW TO RUN:
-----------
        python train.py

After training:
        python main.py --mode yolo --model runs/detect/screw_nut/weights/best.pt

REQUIREMENTS:
        pip install ultralytics
"""

import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] Run:  pip install ultralytics")


# ══════════════════════════════════════════════════════════════
#  ★  EDIT THESE  ★
# ══════════════════════════════════════════════════════════════

# Path to the data.yaml that came with your Roboflow export
DATASET_YAML = "dataset/data.yaml"

# Base model — yolov8n (fastest) → yolov8s → yolov8m (most accurate)
BASE_MODEL = "yolov8s.pt"

# Training settings
EPOCHS     = 100    # more epochs = better accuracy (try 50 first)
IMAGE_SIZE = 640    # keep at 640
BATCH_SIZE = 16     # reduce to 8 if you get memory errors
DEVICE     = 0      # 0 = GPU,  "cpu" = CPU only

# Output folder
PROJECT    = "runs/detect"
RUN_NAME   = "screw_nut"

# ══════════════════════════════════════════════════════════════


def check_dataset():
    """Make sure dataset exists before training."""
    if not os.path.exists(DATASET_YAML):
        print(f"\n[ERROR] Dataset not found at: '{DATASET_YAML}'")
        print("\n  Follow these steps:")
        print("  ① Go to  https://roboflow.com  and create a free account")
        print("  ② Create project → Object Detection")
        print("  ③ Upload images from  dataset_images/  folder")
        print("  ④ Draw boxes & label each object as 'screw' or 'nut'")
        print("  ⑤ Click Generate → Export → YOLOv8 format → Download zip")
        print("  ⑥ Extract the zip into this folder as 'dataset/'")
        print(f"  ⑦ Make sure '{DATASET_YAML}' exists")
        print("  ⑧ Run this script again\n")
        sys.exit(1)
    print(f"[✓] Dataset found : {DATASET_YAML}")


def train():
    check_dataset()

    print(f"\n{'='*55}")
    print(f"  Starting YOLO Training")
    print(f"  Base model  : {BASE_MODEL}")
    print(f"  Dataset     : {DATASET_YAML}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Image size  : {IMAGE_SIZE}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Device      : {'GPU' if DEVICE == 0 else 'CPU'}")
    print(f"{'='*55}\n")

    # Load pretrained base model (downloads automatically ~22MB)
    model = YOLO(BASE_MODEL)

    # Train
    results = model.train(
        data     = DATASET_YAML,
        epochs   = EPOCHS,
        imgsz    = IMAGE_SIZE,
        batch    = BATCH_SIZE,
        device   = DEVICE,
        project  = PROJECT,
        name     = RUN_NAME,

        # ── Data augmentation (improves real-world accuracy) ──
        hsv_h    = 0.015,   # hue shift
        hsv_s    = 0.7,     # saturation
        hsv_v    = 0.4,     # brightness
        degrees  = 45,      # rotation
        flipud   = 0.3,     # vertical flip
        fliplr   = 0.5,     # horizontal flip
        mosaic   = 1.0,     # mosaic augmentation
        mixup    = 0.1,     # mixup augmentation

        # ── Stop early if no improvement ─────────────────────
        patience = 20,
        seed     = 42,
        verbose  = True,
    )

    # ── Show results ──────────────────────────────────────────
    best_model = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"

    print(f"\n{'='*55}")
    print(f"  ✅ Training Complete!")
    print(f"  Best model saved → {best_model}")
    print(f"\n  Run detector with your trained model:")
    print(f"  python main.py --mode yolo --model {best_model}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    train()