"""Использование готовой модели для классификации изображений."""

import requests
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

url = (
    "https://img.freepik.com/premium-photo/"
    "a-house-on-a-mountain-with-a-mountain-"
    "in-the-background_759095-3394.jpg"
)
image = Image.open(requests.get(url, stream=True).raw)

processor = (ViTImageProcessor
             .from_pretrained("google/vit-base-patch16-224"))
model = (ViTForImageClassification
         .from_pretrained("google/vit-base-patch16-224"))

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
