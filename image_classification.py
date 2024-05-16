"""Использование готовой модели для классификации изображений."""

import requests
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# URL изображения
url = (
    "https://img.freepik.com/premium-photo/"
    "a-house-on-a-mountain-with-a-mountain-"
    "in-the-background_759095-3394.jpg"
)
# Используем контекстный менеджер для открытия изображения
with Image.open(requests.get(url, stream=True).raw) as image:

    # Инициализация процессора и модели
    processor = (ViTImageProcessor
                .from_pretrained("google/vit-base-patch16-224"))
    model = (ViTForImageClassification
            .from_pretrained("google/vit-base-patch16-224"))

    # Подготовка изображения для модели
    inputs = processor(images=image, return_tensors="pt")

    # Получение предсказания от модели
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Вывод предсказанного класса
    predicted_class = model.config.id2label[predicted_class_idx]
    print(f"Predicted class: {predicted_class}")
