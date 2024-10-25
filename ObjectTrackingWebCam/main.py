import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # YOLOv8'in en hızlı ve küçük modeli olan 'n' modelini kullanıyoruz

# Webcam akışını başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Frame al
    if not ret:
        break

    # YOLOv8 modeline frame ver ve sonuçları al
    results = model(frame)

    # Sonuçları frame'e çizdir
    annotated_frame = results[0].plot()

    # OpenCV ile sonucu ekranda göster
    cv2.imshow("YOLOv8 Object Tracking", annotated_frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()


