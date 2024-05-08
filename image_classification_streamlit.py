"""Приложение Streamlit для классификации изображений."""

import requests
import streamlit as st
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


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


def load_image(url):
    """Загрузка изображения из указанного URL-адреса
    с помощью библиотеки requests."""
    img = Image.open(requests.get(url, stream=True).raw)
    st.image(img)
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

link = get_image_link()

result = st.button("Распознать изображение")

if result:
    try:
        loaded_image = load_image(link)
        with st.spinner("Идет обработка... Пожалуйста, подождите..."):
            result = image_classification(loaded_image)
        st.markdown(f"Результаты распознавания: :rainbow[{result}]")
        st.snow()
    except IOError:
        st.error(
            "Не удалось найти изображение по указанной ссылке. "
            "Попробуйте снова!",
            icon="😞",
        )
