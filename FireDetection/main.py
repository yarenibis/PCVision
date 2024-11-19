from ultralytics import YOLO
import cv2

# Eğitilmiş modeli yükleyin
model = YOLO("runs/detect/train/weights/best.pt")  # Eğitimden elde edilen model


# Webcam akışını başlat
print("Kamera başlatıldı")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Frame al
    if not ret:
        break

    # YOLOv8 modeline frame ver ve sonuçları al
    results = model(frame, conf=0.6)  # Güven skorunu %60 olarak ayarla

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


