"""
╔══════════════════════════════════════════════════════════════╗
║        SCREW, NUT & BOLT DETECTOR  —  main.py               ║
║        Powered by YOLOv8                                    ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
  python main.py --model runs/detect/screw_nut_bolt/weights/best.pt
  python main.py --model best.pt --cam 1
  python main.py --model best.pt --save

REQUIREMENTS:
    pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] Run:  pip install ultralytics")

# ──────────────────────────────────────────────────────────────
#  CLASS SETTINGS  (index must match your Roboflow label order)
# ──────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "screw": (57,  255,  20),   # neon green
    "nut"  : (0,   165, 255),   # orange
    "bolt" : (0,   80,  255),   # blue
    # fallback for any extra classes
    "default": (200, 200, 200),
}

CONFIDENCE = 0.40
IOU        = 0.40
MODEL_PATH = "best.pt"


# ══════════════════════════════════════════════════════════════
#  OPEN CAMERA
# ══════════════════════════════════════════════════════════════
def _open_camera(cam_index: int):
    print(f"[INFO] Opening webcam {cam_index} ...", end=" ", flush=True)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open webcam {cam_index}. Try --cam 1")
        sys.exit(1)
    print("OK ✓\n")
    return cap


# ══════════════════════════════════════════════════════════════
#  MAIN YOLO LOOP
# ══════════════════════════════════════════════════════════════
def run_yolo(model_path: str, cam_index: int, save: bool = False):

    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model file not found: '{model_path}'")
        print("\n  Options:")
        print("  A) Train your own:")
        print("       python capture.py --label screw")
        print("       python capture.py --label nut")
        print("       python capture.py --label bolt")
        print("       python train.py")
        print(f"\n  B) Use custom path:")
        print(f"       python main.py --model path/to/best.pt")
        sys.exit(1)

    print(f"[✓] Loading model : {model_path}")
    model = YOLO(model_path)
    names = model.names   # {0: 'screw', 1: 'nut', 2: 'bolt'} etc.
    print(f"[✓] Classes       : {list(names.values())}\n")

    cap    = _open_camera(cam_index)
    writer = None

    if save:
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(
            "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"[✓] Recording → output.mp4")

    print("[INFO] Press  Q  or  ESC  to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Lost webcam feed.")
            break

        display = frame.copy()

        # ── Inference ─────────────────────────────────────────
        results = model.predict(
            source  = frame,
            conf    = CONFIDENCE,
            iou     = IOU,
            verbose = False
        )[0]

        # Reset counts for every class the model knows
        counts = {n: 0 for n in names.values()}

        # ── Draw detections ───────────────────────────────────
        for box in results.boxes:
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            name  = names.get(cls_id, f"class_{cls_id}")
            color = CLASS_COLORS.get(name.lower(),
                    CLASS_COLORS["default"])

            counts[name] = counts.get(name, 0) + 1

            # Bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Corner accents
            cl = 14
            th = 3
            cv2.line(display, (x1,y1), (x1+cl,y1), color, th)
            cv2.line(display, (x1,y1), (x1,y1+cl), color, th)
            cv2.line(display, (x2,y1), (x2-cl,y1), color, th)
            cv2.line(display, (x2,y1), (x2,y1+cl), color, th)
            cv2.line(display, (x1,y2), (x1+cl,y2), color, th)
            cv2.line(display, (x1,y2), (x1,y2-cl), color, th)
            cv2.line(display, (x2,y2), (x2-cl,y2), color, th)
            cv2.line(display, (x2,y2), (x2,y2-cl), color, th)

            # Label pill
            label = f"{name}  {confidence:.0%}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            cv2.rectangle(display,
                          (x1, y1 - lh - 12),
                          (x1 + lw + 10, y1),
                          color, -1)
            cv2.rectangle(display,
                          (x1, y1 - lh - 12),
                          (x1 + lw + 10, y1),
                          (0, 0, 0), 1)
            cv2.putText(display, label,
                        (x1 + 5, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (0, 0, 0), 2)

        # ── HUD ───────────────────────────────────────────────
        _draw_hud(display, counts, names)

        cv2.imshow("Screw / Nut / Bolt Detector  [YOLOv8]", display)

        if writer:
            writer.write(display)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    if writer:
        writer.release()
        print("\n[✓] Saved → output.mp4")
    cv2.destroyAllWindows()
    print("[✓] Done.")


# ══════════════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════════════
def _draw_hud(frame, counts: dict, class_names: dict):
    n       = len(class_names)
    pad     = 14
    line_h  = 38
    panel_h = pad + n * line_h + pad
    panel_w = 260

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (panel_w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    cv2.rectangle(frame, (8, 8), (panel_w, panel_h), (80, 80, 80), 1)

    for i, (cls_id, name) in enumerate(class_names.items()):
        color = CLASS_COLORS.get(name.lower(), CLASS_COLORS["default"])
        count = counts.get(name, 0)
        y     = pad + (i + 1) * line_h

        cv2.circle(frame, (26, y - 6), 7, color, -1)
        cv2.putText(frame,
                    f"{name:<8}:  {count}",
                    (42, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, 2)

        if i < n - 1:
            cv2.line(frame, (14, y + 10), (panel_w - 6, y + 10),
                     (50, 50, 50), 1)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
def main():
    global CONFIDENCE
    p = argparse.ArgumentParser(
        description="Screw / Nut / Bolt Detector — YOLOv8")

    p.add_argument("--model",
                   default=MODEL_PATH,
                   help=f"Path to YOLOv8 .pt model  (default: {MODEL_PATH})")
    p.add_argument("--cam",
                   type=int, default=0,
                   help="Webcam index  (default: 0)")
    p.add_argument("--conf",
                   type=float, default=CONFIDENCE,
                   help=f"Confidence threshold  (default: {CONFIDENCE})")
    p.add_argument("--save",
                   action="store_true",
                   help="Save output to output.mp4")

    args       = p.parse_args()
    CONFIDENCE = args.conf

    print("=" * 48)
    print("   SCREW / NUT / BOLT DETECTOR  [YOLOv8]")
    print(f"   Model      : {args.model}")
    print(f"   Camera     : Webcam {args.cam}")
    print(f"   Confidence : {CONFIDENCE}")
    print("=" * 48 + "\n")

    run_yolo(args.model, args.cam, save=args.save)


if __name__ == "__main__":
    main()