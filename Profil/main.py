import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from ultralytics import YOLO

# YOLOv8 model dosyasını yükle
model = YOLO("yolov8n.pt")

# URL'lerin bulunduğu dosyayı okuma
url_file = "dosya.txt"
with open(url_file, "r") as f:
    urls = [line.strip().replace('"', '') for line in f.readlines()]

# Sonuçları tutmak için bir liste
results = []

for url in urls:
    if not url.startswith("http"):
        print(f"Görsel bulunamadı: {url}")
        results.append((url, "Geçersiz URL"))
        continue

    try:
        response = requests.get(url)
        html_content = response.text # İsteğin yanıtını response değişkenine atar ve yanıtın metin içeriğini (html_content) alır.

        # BeautifulSoup ile alınan HTML içeriğini ayrıştırır. img etiketlerini bulur
        # ve bu etiketlerin src özniteliklerini bir listeye ekler.
        soup = BeautifulSoup(html_content, "html.parser")
        img_tags = soup.find_all("img")
        img_urls = [img['src'] for img in img_tags if 'src' in img.attrs] # Bu yapı, img_tags listesindeki
         # her bir img etiketinin src özniteliğini alarak yeni bir liste oluşturur.

        # Göreceli URL'yi tam URL'ye dönüştür
        # Eğer en az bir görsel URL'si varsa, ilk görselin URL'sini alır ve konsola yazdırır.
        # Sonra bu göreceli URL'yi tam URL'ye dönüştürür ve bu tam URL'ye bir başka HTTP isteği gönderir.
        if img_urls:
            image_url = img_urls[0]
            print(f"Görsel URL: {image_url}")
            full_image_url = urljoin(url, image_url)
            image_response = requests.get(full_image_url)


            #Görselin içeriğini byte dizisine çevirir ve OpenCV ile bir görüntü olarak yükler.
            #Eğer görüntü yüklenememişse, bir hata mesajı basar ve döngünün bir sonraki iterasyonuna geçer.
            image = np.asarray(bytearray(image_response.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if image is None:
                print(f"Görüntü yüklenemedi: {url}")
                results.append((url, "İndirilemedi"))
                continue

        else:
            print(f"Görsel bulunamadı: {url}")
            results.append((url, "Görsel bulunamadı"))
            continue

    except Exception as e:
        print(f"Görüntü indirilemedi: {url}, Hata: {e}")
        results.append((url, "İndirilemedi"))
        continue

    # YOLOv8 ile tespit yap
    detections = model.predict(image)

    person_detected = False

    # Tespit edilen her nesne için döngü başlatır. Eğer tespit edilen nesne "person" ise ve
    # güven skoru (confidence) %60'tan fazlaysa, person_detected değişkenini True yapar ve
    # tespit edilen nesnenin etrafına bir dikdörtgen çizer. Ayrıca, üzerine "Person" yazısını ekler.
    for box in detections[0].boxes:
        class_id = int(box.cls[0])
        confidence = box.conf[0]
        if model.names[class_id] == "person" and confidence > 0.6:
            person_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinatları
            # al-x1, y1: Dikdörtgenin sol üst köşesinin (başlangıç noktası) koordinatlarıdır.
            #x2, y2: Dikdörtgenin sağ alt köşesinin (bitiş noktası) koordinatlarıdır.
            #map(int, ...): Bu, box.xyxy[0] değerlerini tam sayılara (integer) dönüştürmek için kullanılır.
            #Algılama sonucundaki koordinatlar genellikle ondalıklı sayılar olarak gelir; int fonksiyonu bu değerleri tam sayılara dönüştürür.
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) #(0, 255, 0): Çizilecek dikdörtgenin rengi (bu durumda yeşil, BGR formatında).
            cv2.putText(image, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Sonucu kaydet
    if person_detected:
        results.append((url, "İnsan"))
    else:
        results.append((url, "İnsan Değil"))

    # Resmi ekranda gösterme
    cv2.imshow(f"Sonuç: {url}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Sonuçları yazdırma
for url, result in results:
    print(f"{url}: {result}")


