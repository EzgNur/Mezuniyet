from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import numpy as np
from PIL import Image, ImageOps

# 'groups' parametresini kaldırarak DepthwiseConv2D yükleme fonksiyonu
def custom_depthwise_conv2d(**kwargs):
    kwargs.pop("groups", None)
    return DepthwiseConv2D(**kwargs)

# Modeli yükle
model = load_model("/content/keras_model.h5", custom_objects={"DepthwiseConv2D": custom_depthwise_conv2d})

# Load the labels
class_names = [line.strip() for line in open("/content/labels.txt", "r").readlines()]

# Resmi oku ve ön işle
image = Image.open("/content/drive/MyDrive/VeriSeti/Farı Açık Araba Görselleri/1.jpeg").convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

# Tahmin yap
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]  # Doğrudan etiketi al
confidence_score = prediction[0][index]

# Tahmini ekrana yazdır
print("Class:", class_name)
print("Confidence Score:", confidence_score)

# Öneri verme
if "Class 1" in class_name:
    print("🌍 İklim değişikliği tespit edildi:")
    print("- Hayvanların yaşam alanına uygun bir yer değil .")
    print("- Bunu düzeltmek için iklim değişikliğinin önüne geçebiliriz.")
    print("- Geri dönüşümü artırın ve plastik kullanımını azaltın.")
elif "Class 2" in class_name:
    print("✅ İklim değişikliği belirtileri tespit edilmedi. Ancak çevreyi korumaya devam edin!")
    print("- Ağaç dikin ve yeşil alanları koruyun.")
    print("- Doğal kaynakları israf etmeyin, enerji tasarrufu yapın.")
    print("- Çevre dostu ürünler kullanın ve sürdürülebilir yaşamı destekleyin.")
else:
    print("⚠️ Model, iklim değişikliği ile ilgili bir sınıf belirleyemedi.")
