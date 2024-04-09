"""Готовая модель классификации изображений с streamlit."""

from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
import streamlit as st
from PIL import Image
import requests


@st.cache_resource
def load_model():
    """Загрузка модели для классификации изображений."""
    return (ViTForImageClassification
            .from_pretrained('google/vit-base-patch16-224'))


@st.cache_resource
def load_processor():
    """Загрузка процессора для пред. изображений в требуемом формате."""
    return ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')


def get_image_link():
    """Отображение текс. поля для ввода ссылки на изображение."""
    return st.text_input("Введите ссылку на изображение для распознавания")


def load_image(url):
    """Получение изображения и вывод его на экран."""
    img = Image.open(requests.get(url, stream=True).raw)
    st.image(img)
    return img


def image_classification(picture):
    """Обработка и распознавание изображения."""
    inputs = processor(images=picture, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


def show_results(results):
    """Вывод результатов на экран."""
    st.write(results)


processor = load_processor()
model = load_model()

st.title('Модель для классификации изображений vit-base-patch16-224')

link = get_image_link()

result = st.button('Распознать изображение')
"""
Обработка исключений, которые приведут к ошибке в случае отсутствия ссылки
или указания ссылки на объект, который не является изображением.
"""
if result:
    try:
        loaded_image = load_image(link)
        with st.spinner('Идет обработка... Пожалуйста, подождите...'):
            result = image_classification(loaded_image)
        st.markdown(f'Результаты распознавания: :rainbow[{result}]')
        st.snow()
    except IOError:
        st.error(
            """
            Не удалось найти изображение по указанной ссылке.
            Попробуйте снова!
            """,
            icon="😞")
