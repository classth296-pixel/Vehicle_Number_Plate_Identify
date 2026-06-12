# Vehicle License Plate Detection using YOLOv8

I built this project to detect vehicle license plates in real time using YOLOv8. It works on images, and also on live webcam feed. Nothing fancy — just a clean detection pipeline that actually runs.

---

## What this does

- Detects license plates in images
- Works live with a webcam
- Trained on a custom dataset using YOLOv8 nano (fast, works on low VRAM GPUs)

---

## Project Structure

```
License_plate/
│
├── data.yaml          # tells YOLOv8 where your dataset is
├── main.py            # run this to train the model
├── predict.py         # run this to test on an image
├── webcam.py          # run this for live webcam detection
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── test_images/       # put your test images here
│
└── runs/              # YOLOv8 saves trained weights here automatically
    └── detect/
        └── train-10/
            └── weights/
                ├── best.pt   ← use this for prediction
                └── last.pt
```

---

## Setup

You just need two libraries:

```bash
pip install ultralytics opencv-python
```

A GPU helps but isn't required. The model will use CUDA automatically if available.

---

## Dataset

Labels are in YOLO format — one `.txt` file per image.

Each line in the label file:
```
<class_id> <x_center> <y_center> <width> <height>
```

The `data.yaml` file looks like this:
```yaml
path: License_plate
train: dataset/images/train
val: dataset/images/val

nc: 1
names: ['license_plate']
```

---

## Training

```python
# main.py
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(
        data=r'License_plate\data.yaml',
        epochs=5,
        batch=10,
        imgsz=64,
        workers=0       # keep this 0 on Windows to avoid errors
    )
```

```bash
python main.py
```

After training, the best weights get saved to `runs/detect/trainX/weights/best.pt`.

---

## Predict on an Image

```python
# predict.py
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs\detect\train-10\weights\best.pt')
    results = model(r'License_plate\test_images\sample.jpg')
    results[0].show()                        # opens a popup
    results[0].save(filename='output.jpg')  # or save to file
```

```bash
python predict.py
```

---

## Live Webcam Detection

```python
# webcam.py
from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO(r'runs\detect\train-10\weights\best.pt')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow('License Plate Detection', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

```bash
python webcam.py
```

Press **Q** to quit.

---

## A few things I noticed

- `imgsz=64` is very small — bumping it to 320 or 640 gives noticeably better results
- More epochs (50+) and a bigger dataset will improve accuracy a lot
- `workers=0` is needed on Windows, otherwise multiprocessing throws errors

---

## Built with

- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- OpenCV

---
