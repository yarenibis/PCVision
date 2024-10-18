from ultralytics import YOLO
model=YOLO('yolov8n.pt')


sonuc=model.predict(source='car-detection.mp4', show=True)
