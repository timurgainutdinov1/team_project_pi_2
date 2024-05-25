"Загрузка и работа с моделью и процессором."

from transformers import ViTForImageClassification, ViTImageProcessor


def load_model():
    """Загрузка модели."""
    return ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")


def load_processor():
    """Загрузка процессора для обработки изображений."""
    return ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


def image_classification(picture, processor, model):
    """Обработка и распознавание изображения."""
    inputs = processor(images=picture, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]
