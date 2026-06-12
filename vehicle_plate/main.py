from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    train_data = model.train(
        data=r'License_plate\data.yaml',
        epochs=5,
        batch=10,
        imgsz=64,
        workers=0
    )
