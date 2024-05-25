"Загрузка и работа с моделью и процессором."

from transformers import ViTForImageClassification, ViTImageProcessor


def load_model():
    """Загрузка модели."""
    return ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")


def load_processor():
    """Загрузка процессора для обработки изображений."""
    return ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
