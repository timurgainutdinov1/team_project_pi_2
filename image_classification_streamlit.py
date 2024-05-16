"""Приложение Streamlit для классификации изображений."""

import requests
import streamlit as st
import translators as ts
from PIL import Image, UnidentifiedImageError
from requests.exceptions import MissingSchema
from transformers import ViTForImageClassification, ViTImageProcessor


class MissingSourceError(Exception):
    """Класс представляет ошибку,
    возникающую при отсутствии
    источника изображения."""
    pass


class TwoSourcesError(Exception):
    """Класс представляет ошибку,
    возникающую при указании
    двух источников изображений."""
    pass


@st.cache_resource
def load_model():
    """Загрузка модели"""
    return (ViTForImageClassification
            .from_pretrained("google/vit-base-patch16-224"))


@st.cache_resource
def load_processor():
    """Загрузка процессора для обработки изображений."""
    return (ViTImageProcessor
            .from_pretrained("google/vit-base-patch16-224"))


def get_image_link():
    """Ввод URL-адреса с изображением."""
    return st.text_input("Введите ссылку на изображение для распознавания")


def get_image_file():
    """Загрузка файла с изображением."""
    return st.file_uploader("Или загрузите изображение из файла")


def load_image_from_url(url):
    """Загрузка изображения из указанного URL-адреса
    с помощью библиотеки requests."""
    img = Image.open(requests.get(url, stream=True).raw)
    return img


def load_image_from_file(file):
    """Загрузка изображения из файла."""
    img = Image.open(file)
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


def show_results(results):
    """Вывод результатов"""
    st.write(results)


processor = load_processor()
model = load_model()

st.title("Модель для классификации изображений vit-base-patch16-224")

image_link = get_image_link()
image_file = get_image_file()

result = st.button("Распознать изображение")

if result:
    try:
        loaded_image = ""
        if image_link != "" and image_file is not None:
            raise TwoSourcesError
        elif image_link != "":
            loaded_image = load_image_from_url(image_link)
        elif image_file is not None:
            loaded_image = load_image_from_file(image_file)
        else:
            raise MissingSourceError
        st.image(loaded_image)
        with st.spinner("Идет обработка... Пожалуйста, подождите..."):
            result = image_classification(loaded_image)
        translated_result = ts.translate_text(result,
                                              translator="bing",
                                              from_language="en",
                                              to_language="ru")
        st.markdown(f"Результаты распознавания: {translated_result}")
    except MissingSourceError:
        st.error(
            "Вы не предоставили источник "
            "для загрузки изображения. "
            "Загрузите файл с изображением или укажите ссылку "
            "и попробуйте снова!",
            icon="😞",
        )
    except MissingSchema:
        st.error(
            "Некорректная ссылка! "
            "Укажите корректную ссылку "
            "и попробуйте снова!",
            icon="😞",
        )
    except UnidentifiedImageError:
        st.error(
            "Ваша ссылка или файл не содержат изображения. "
            "Предоставьте корректную ссылку или файл "
            "и попробуйте снова!",
            icon="😞",
        )
    except TwoSourcesError:
        st.error(
            "Вы указали два источника. "
            "Удалите один из источников "
            "и попробуйте снова!",
            icon="😞",
        )