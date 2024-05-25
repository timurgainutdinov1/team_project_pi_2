"Вспомогательные функции для обработки изображений и работы с моделью."

import requests
from PIL import Image
import translators as ts


def load_image_from_url(url):
    """Загрузка изображения из указанного URL-адреса."""
    img = Image.open(requests.get(url, stream=True).raw)
    return img


def load_image_from_file(file):
    """Загрузка изображения из файла."""
    img = Image.open(file)
    return img


def translate_text(text, from_language, to_language):
    """Перевод текста с одного языка на другой."""
    return ts.translate_text(text, translator="bing", from_language=from_language, to_language=to_language)
