"""
╔══════════════════════════════════════════════════════════════╗
║              DATASET CAPTURE  —  capture.py                 ║
║   Captures images from webcam for YOLO training             ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
-----------
  Step 1 — Capture SCREW images:
        python capture.py --label screw

  Step 2 — Capture NUT images:
        python capture.py --label nut

  Step 3 — Capture BOLT images:
        python capture.py --label bolt

CONTROLS (while camera window is open):
  S        → Save current frame (one photo)
  SPACE    → Auto-capture every 1 second (toggle on/off)
  Q / ESC  → Quit

TIPS:
  • Capture at least 100-200 images per class
  • Vary angles: top view, side view, slight tilt
  • Vary backgrounds: white paper, dark cloth, table
  • Vary lighting: bright, dim, with shadows
  • Mix close-up and zoomed-out shots
"""

import cv2
import os
import time
import argparse

# ──────────────────────────────────────────────────────────────
SAVE_DIR  = "dataset_images"
CAM_INDEX = 0


def capture(label: str):
    save_path = os.path.join(SAVE_DIR, label)
    os.makedirs(save_path, exist_ok=True)

    existing = len([f for f in os.listdir(save_path) if f.endswith(".jpg")])
    count    = existing

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Try changing CAM_INDEX to 1.")
        return

    # Label colours
    color_map = {
        "screw": (57,  255,  20),
        "nut"  : (0,   165, 255),
        "bolt" : (0,   80,  255),
    }
    color = color_map.get(label, (255, 255, 255))

    auto_capture   = False
    last_auto_time = 0
    AUTO_INTERVAL  = 1.0

    print(f"\n{'='*50}")
    print(f"  Capturing images for  →  {label.upper()}")
    print(f"  Saving to             →  {save_path}/")
    print(f"  Already saved         →  {existing} images")
    print(f"{'='*50}")
    print(f"  S      = Save one image")
    print(f"  SPACE  = Toggle auto-capture (every {AUTO_INTERVAL}s)")
    print(f"  Q/ESC  = Quit")
    print(f"{'='*50}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        now     = time.time()

        # Auto-capture
        if auto_capture and (now - last_auto_time) >= AUTO_INTERVAL:
            filename = os.path.join(save_path, f"{label}_{count:04d}.jpg")
            cv2.imwrite(filename, frame)
            count         += 1
            last_auto_time = now
            print(f"  [AUTO] Saved: {filename}  (total: {count})")

        # Status overlay
        status_color = (0, 255, 0) if not auto_capture else (0, 100, 255)
        status_text  = "AUTO ON" if auto_capture else "MANUAL"

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 90), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        cv2.putText(display,
                    f"Class: {label.upper()}   Saved: {count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display,
                    f"Mode: {status_text}   |   S=Save  SPACE=Auto  Q=Quit",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

        # Green flash on save
        if auto_capture and (now - last_auto_time) < 0.15:
            cv2.rectangle(display, (0, 0),
                          (display.shape[1], display.shape[0]),
                          (0, 255, 0), 6)

        cv2.imshow(f"Capture — {label.upper()}", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            filename = os.path.join(save_path, f"{label}_{count:04d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"  [SAVED] {filename}  (total: {count})")

        elif key == ord(" "):
            auto_capture = not auto_capture
            state = "ON  ← capturing every 1 second" if auto_capture else "OFF"
            print(f"  [AUTO-CAPTURE] {state}")

        elif key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"  Done!  Total images saved: {count}")
    print(f"  Folder: {os.path.abspath(save_path)}")

    next_steps = {"screw": "nut", "nut": "bolt", "bolt": None}
    nxt = next_steps.get(label)
    if nxt:
        print(f"\n  ✅ Now run:  python capture.py --label {nxt}")
    else:
        print(f"\n  ✅ All 3 classes captured!")
        print(f"  ✅ Upload images from '{SAVE_DIR}/' to Roboflow")
        print(f"     https://roboflow.com")
    print(f"{'='*50}\n")


def main():
    global CAM_INDEX
    p = argparse.ArgumentParser(description="Capture training images")
    p.add_argument("--label",
                   choices=["screw", "nut", "bolt"],
                   required=True,
                   help="Class to capture:  screw | nut | bolt")
    p.add_argument("--cam",
                   type=int,
                   default=CAM_INDEX,
                   help="Webcam index (default: 0)")
    args      = p.parse_args()
    CAM_INDEX = args.cam
    capture(args.label)


if __name__ == "__main__":
    main()