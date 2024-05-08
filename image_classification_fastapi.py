"""Модель классификации изображений с FastAPI."""

import requests
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import ViTForImageClassification, ViTImageProcessor


class ImageRequest(BaseModel):
    """Класс запроса для обработки изображения."""

    url: str


app = FastAPI()

# Процессор для представления изображений в требуемом формате
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Модель для классификации изображений
model = (ViTForImageClassification
         .from_pretrained("google/vit-base-patch16-224"))


def load_image(url):
    """Загрузка изображения из указанного URL-адреса
    с помощью библиотеки requests."""
    img = Image.open(requests.get(url, stream=True).raw)
    return img


def image_classification(picture):
    """Обработка и распознавание изображения.

    Принимает изображение, преобразует его в требуемый формат
    с помощью процессора, пропускает его через модель,
    получает вероятности классов и возвращает предсказанный класс.
    """
    inputs = processor(images=picture, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


@app.get("/")
def root():
    """Маршрут для корневого URL-адреса.

    Возвращает сообщение, указывающее, что это API классификации изображений.
    """
    return {"message": "Image classification API"}


@app.post("/classify-image")
def classify_image(request: ImageRequest):
    """Классифицирует изображение с помощью готовой модели ViT.

    Принимает запрос с URL-адресом изображения, загружает изображение,
    классифицирует его с помощью готовой модели ViT и возвращает результат.
    Если изображение не может быть загружено, возвращается сообщение об ошибке.
    """
    try:
        loaded_image = load_image(request.url)
        result = image_classification(loaded_image)
        return {"result": result}
    except IOError:
        return {"error": "Failed to load image from the provided URL"}
