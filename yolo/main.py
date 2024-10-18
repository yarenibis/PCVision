from ultralytics import YOLO
from PIL import Image

#model yükleme
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

img= Image.open('Elon_Musk_Royal_Society.jpg')
sonuc=model.predict(source= img, save=True)
