import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntü dosyasını yükle
image_path = "D:\\PycharmProjects\\WordShapes\\Shapes.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")

# Sobel operatörü kullanarak yatay (x) ve dikey (y) türevlerini bulma
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Yatay türev (x yönü)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Dikey türev (y yönü)

# Mutlak değerlerini alarak negatif kenarları pozitif yapma
sobel_x = np.abs(sobel_x)
sobel_y = np.abs(sobel_y)

# Kenarları birleştirerek daha güçlü bir kenar görüntüsü oluşturma
combined_sobel = np.sqrt(sobel_x*2 + sobel_y*2)

# Kenarları daha iyi göstermek için normalize etme (0-255 aralığına)
combined_sobel = (combined_sobel / combined_sobel.max()) * 255
combined_sobel = combined_sobel.astype(np.uint8)

# Sonuçları gösterme
plt.figure(figsize=(15, 5))

# Orijinal görüntü
plt.subplot(1, 3, 1)
plt.title("Orijinal Görüntü")
plt.imshow(image, cmap='gray')
plt.axis("off")

# Yatay türev sonucu (Sobel X)
plt.subplot(1, 3, 2)
plt.title("Yatay Türev (Sobel X)")
plt.imshow(sobel_x, cmap='gray')
plt.axis("off")

# Dikey türev sonucu (Sobel Y)
plt.subplot(1, 3, 3)
plt.title("Dikey Türev (Sobel Y)")
plt.imshow(sobel_y, cmap='gray')
plt.axis("off")

# Kenarları birleştirerek gösterme
plt.figure(figsize=(5, 5))
plt.title("Birleştirilmiş Türev (Sobel X + Y)")
plt.imshow(combined_sobel, cmap='gray')
plt.axis("off")

plt.show()
