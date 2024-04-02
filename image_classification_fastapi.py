from PIL import Image
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import ViTImageProcessor, ViTForImageClassification


class ImageRequest(BaseModel):
    url: str


app = FastAPI()
# Процессор для представления изображений в требуемом формате
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# Модель для классификации изображений
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


# Получение изображения
def load_image(url):
    img = Image.open(requests.get(url, stream=True).raw)
    return img


# Обработка и распознавание изображения
def image_classification(picture):
    inputs = processor(images=picture, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


# Маршрут для корневого URL-адреса
@app.get("/")
def root():
    return {"message": "Image classification API"}


@app.post("/classify-image")
def classify_image(request: ImageRequest):
    """
    Classify an image using a pre-trained ViT model.
    """
    try:
        loaded_image = load_image(request.url)
        result = image_classification(loaded_image)
        return {"result": result}
    except IOError:
        return {"error": "Failed to load image from the provided URL"}
