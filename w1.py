"""
╔══════════════════════════════════════════════════════════════╗
║           SCREW & NUT DETECTOR  —  main.py                  ║
║   Nut   → compact / near-circular / hexagonal shape         ║
║   Screw → elongated shape at any angle                      ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
-----------
  python main.py                        ← OpenCV mode (no model needed)
  python main.py --cam 1                ← different webcam
  python main.py --mode yolo --model best.pt

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
    YOLO_READY = True
except ImportError:
    YOLO_READY = False

# ──────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Screw": (57,  255,  20),   # neon green
    "Nut"  : (0,   165, 255),   # orange
}
CONFIDENCE = 0.45
IOU        = 0.40


# ══════════════════════════════════════════════════════════════
#  OPEN CAMERA
# ══════════════════════════════════════════════════════════════
def _open_camera(cam_index: int):
    print(f"[INFO] Opening webcam {cam_index} ...", end=" ", flush=True)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open webcam {cam_index}.")
        print("        Try:  python main.py --cam 1")
        sys.exit(1)
    print("OK ✓\n")
    return cap


# ══════════════════════════════════════════════════════════════
#  MODE 1 ─ YOLO
# ══════════════════════════════════════════════════════════════
def run_yolo(model_path: str, cam_index: int, save: bool = False):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: '{model_path}'")
        print("        Train first → python train.py")
        print("        Switching to OpenCV mode...\n")
        run_opencv(cam_index)
        return

    print(f"[✓] YOLO model : {model_path}")
    model  = YOLO(model_path)
    cap    = _open_camera(cam_index)
    writer = None

    if save:
        w, h = int(cap.get(3)), int(cap.get(4))
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(
            "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("[INFO] Press  Q  or  ESC  to quit")
    CLASS_NAMES = {0: "Screw", 1: "Nut"}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame, conf=CONFIDENCE, iou=IOU, verbose=False)[0]

        display = frame.copy()
        counts  = {"Screw": 0, "Nut": 0}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name  = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
            color = CLASS_COLORS.get(name, (200, 200, 200))
            if name in counts:
                counts[name] += 1
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(display, (x1, y1-lh-10), (x1+lw+6, y1), color, -1)
            cv2.putText(display, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        _draw_hud(display, counts["Screw"], counts["Nut"])
        cv2.imshow("Screw & Nut Detector [YOLO]", display)
        if writer:
            writer.write(display)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════
#  MODE 2 ─ OPENCV  (handles ALL screw orientations)
# ══════════════════════════════════════════════════════════════
def run_opencv(cam_index: int = 0):
    print("[INFO] Running OpenCV detector")
    print("[INFO] Detects screws at ANY angle (horizontal, vertical, tilted)")
    print("[INFO] Press  Q  or  ESC  to quit\n")

    cap = _open_camera(cam_index)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # ── Pre-process ───────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # ── Detect objects (circles + contours) ───────────────
        detections  = _detect_all(gray, frame)
        detections  = _nms(detections, iou_thresh=0.35)

        counts = {"Screw": 0, "Nut": 0}

        for (label, x1, y1, x2, y2) in detections:
            color = CLASS_COLORS[label]
            counts[label] += 1

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display,
                          (x1, y1 - lh - 8), (x1 + lw + 6, y1),
                          color, -1)
            cv2.putText(display, label,
                        (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2)

        _draw_hud(display, counts["Screw"], counts["Nut"])
        cv2.imshow("Screw & Nut Detector [OpenCV]", display)

        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✓] Done.")


# ══════════════════════════════════════════════════════════════
#  MAIN DETECTOR  — shape-based classification
# ══════════════════════════════════════════════════════════════
def _detect_all(gray, frame):
    """
    Single contour pipeline — classifies each object by SHAPE:

    NUT   → compact, near-square bounding box, aspect ratio close to 1
            (hex nuts and round nuts both satisfy this)

    SCREW → elongated bounding box, aspect ratio > 2.0
            Works at any angle: horizontal, vertical, diagonal
    """
    h_f, w_f = frame.shape[:2]

    # ── Edge map ──────────────────────────────────────────────
    blur  = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 25, 90)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    edges  = cv2.dilate(edges, kernel, iterations=2)
    edges  = cv2.erode(edges,  kernel, iterations=1)

    # ── Contours ──────────────────────────────────────────────
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter by area — ignore noise and background
        if area < 500 or area > 100_000:
            continue

        # ── Rotated bounding rectangle ────────────────────────
        rect           = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect

        # Always make rw the longer side
        if rw < rh:
            rw, rh = rh, rw

        if rh < 1:
            continue

        aspect_ratio  = rw / rh          # elongation: 1.0 = square, >2 = rod
        extent        = area / (rw * rh + 1e-5)  # solidity in bounding box

        # ── Circularity (extra cue for nuts) ──────────────────
        perimeter   = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-5))
        # Perfect circle = 1.0, elongated rod → near 0

        # ── Axis-aligned box for drawing ──────────────────────
        box_pts = np.int32(cv2.boxPoints(rect))
        x1 = int(max(0,   np.min(box_pts[:, 0])))
        y1 = int(max(0,   np.min(box_pts[:, 1])))
        x2 = int(min(w_f, np.max(box_pts[:, 0])))
        y2 = int(min(h_f, np.max(box_pts[:, 1])))

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        # ── CLASSIFICATION ────────────────────────────────────
        #
        #  NUT:   compact shape → aspect_ratio < 2.0
        #                         circularity  > 0.45
        #                         extent       > 0.50
        #
        #  SCREW: rod-like shape → aspect_ratio > 2.0
        #                          circularity  < 0.55
        #

        if aspect_ratio < 2.0 and circularity > 0.45 and extent > 0.45:
            results.append(("Nut",   x1, y1, x2, y2))

        elif aspect_ratio >= 2.0 and circularity < 0.55 and extent > 0.25:
            results.append(("Screw", x1, y1, x2, y2))

    return results


# ══════════════════════════════════════════════════════════════
#  NMS — remove overlapping boxes
# ══════════════════════════════════════════════════════════════
def _nms(detections, iou_thresh=0.35):
    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda d: (d[3] - d[1]) * (d[4] - d[2]),
        reverse=True
    )

    kept = []
    used = [False] * len(detections)

    for i, d1 in enumerate(detections):
        if used[i]:
            continue
        kept.append(d1)
        for j, d2 in enumerate(detections):
            if i == j or used[j]:
                continue
            if _iou(d1[1:], d2[1:]) > iou_thresh:
                used[j] = True

    return kept


def _iou(box1, box2):
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


# ══════════════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════════════
def _draw_hud(frame, screws: int, nuts: int):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (235, 115), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.line(frame, (8, 57), (235, 57), (60, 60, 60), 1)
    cv2.putText(frame, f"Screws :  {screws}",
                (18, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, CLASS_COLORS["Screw"], 2)
    cv2.putText(frame, f"Nuts   :  {nuts}",
                (18, 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, CLASS_COLORS["Nut"], 2)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Screw & Nut Detector")
    p.add_argument("--cam",   type=int, default=0,
                   help="Webcam index (default: 0)")
    p.add_argument("--mode",  choices=["opencv", "yolo"], default="opencv",
                   help="Detection engine (default: opencv)")
    p.add_argument("--model", default="best.pt",
                   help="YOLOv8 model path (default: best.pt)")
    p.add_argument("--save",  action="store_true",
                   help="Save output to output.mp4")
    args = p.parse_args()

    print("=" * 45)
    print("   SCREW & NUT DETECTOR")
    print(f"   Camera : Webcam index {args.cam}")
    print(f"   Mode   : {args.mode.upper()}")
    print("=" * 45 + "\n")

    if args.mode == "yolo":
        if not YOLO_READY:
            print("[!] pip install ultralytics → switching to OpenCV\n")
            run_opencv(args.cam)
        else:
            run_yolo(args.model, args.cam, save=args.save)
    else:
        run_opencv(args.cam)


if __name__ == "__main__":
    main()