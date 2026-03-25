# 🔍 Object Detection using YOLOv8

This project detects **screw, nut, and bolt** using a custom-trained YOLOv8 model.

---

## 🚀 Features

* Real-time object detection
* Custom dataset trained on Roboflow
* Supports webcam detection
* YOLOv8 model (Ultralytics)

---

## 🧠 Classes

* Screw
* Nut
* Bolt

---

## 🛠️ Tech Stack

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* Roboflow

---

## 📂 Project Structure

```
object-detection-yolo/
│
├── capture.py        # Capture images
├── train.py          # Train model
├── main.py           # Run detection
├── dataset_images/   # Raw images
├── dataset/          # YOLO dataset (ignored)
└── runs/             # Training results (ignored)
```

---

## ⚙️ Installation

```bash
pip install ultralytics opencv-python
```

---

## ▶️ How to Run

### 1. Capture images

```bash
python capture.py --label screw
```

### 2. Train model

```bash
python train.py
```

### 3. Run detection

```bash
python main.py --model runs/detect/screw_nut_bolt/weights/best.pt
```

---

## 📸 Demo

(Add your screenshots or video here later)

---

## ⚠️ Notes

* Dataset and model weights are not included in repo
* Train your own model using Roboflow export

---

## 👨‍💻 Author

**Logesh S**
