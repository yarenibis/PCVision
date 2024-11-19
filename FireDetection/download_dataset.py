import torch
from ultralytics import YOLO
from roboflow import Roboflow

print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv8 modelini başlat
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli

rf = Roboflow(api_key="pH0Ct44vA1l4cErsDav8")
project = rf.workspace("roboflow-universe-projects").project("fire-and-smoke-segmentation")
version = project.version(2)
dataset = version.download("yolov8")  # veri setini indirip yolov8 formatına göre hazırlar


model.train(data=f"{dataset.location}/data.yaml", epochs=20)
# Data.yaml: modelin eğitimde kullanacağı sınıfları ve veri yolunu içerir.










