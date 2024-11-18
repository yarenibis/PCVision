import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # YOLOv8'in en hızlı ve küçük modeli olan 'n' modelini kullanıyoruz

#Disteki görüntüyü sınıflandırma
image=Image.open('images.jpg')
model.predict(source=image, save=True)
print("diskteki görüntü sınıflandırıldı")

#Url'deki görüntüyü sınıflandırma
# Ana sayfa URL'si
page_url = "https://edition.cnn.com/2024/10/31/politics/us-north-korean-troops-combat-ukraine/index.html"  # Görsellerin bulunduğu sayfanın URL'sini buraya ekleyin
response = requests.get(page_url)
soup = BeautifulSoup(response.content, "html.parser")

# Sayfadaki img etiketlerinden URL'leri bul
img_tags = soup.find_all("img")
results = []

for img_tag in img_tags:
    img_url = img_tag.get("src")
    if img_url:
        full_img_url = urljoin(page_url, img_url)  # Göreceli URL'yi tam URL'ye çevir
    else:
        print("Geçersiz URL")
        continue

    try:
        image_response = requests.get(full_img_url)
        image = np.asarray(bytearray(image_response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Görüntü yüklenemedi: {full_img_url}")
            results.append((full_img_url, "İndirilemedi"))
            continue

    except Exception as e:
        print(f"Görüntü indirilemedi: {full_img_url}, Hata: {e}")
        results.append((full_img_url, "İndirilemedi"))
        continue

    # YOLOv8 ile tespit yap
    detections = model.predict(image, verbose=False)  # Herhangi bir sınıfı tespit et

    detected_objects = []
    # Tespit edilen her nesneyi listele ve ekrana çiz
    for box in detections[0].boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]
        detected_objects.append(object_name)

        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinatları
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dikdörtgen çizer
        cv2.putText(image, object_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Sonucu kaydet ve ekrana göster
    results.append((full_img_url, ", ".join(detected_objects) if detected_objects else "Nesne Yok"))
    cv2.imshow(f"Sonuç: {full_img_url}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Webcam akışını başlat
print("Kamera başlatıldı")
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


