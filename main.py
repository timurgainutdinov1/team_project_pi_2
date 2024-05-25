"Файл для запуска приложения streamlit."

import streamlit as st
from PIL import UnidentifiedImageError
from requests.exceptions import MissingSchema
from utils import load_image_from_url, load_image_from_file, translate_text
from model import load_model, load_processor, image_classification
from exceptions import MissingSourceError, TwoSourcesError


def get_image_link():
    """Ввод URL-адреса с изображением."""
    return st.text_input("Введите ссылку на изображение для распознавания")


def get_image_file():
    """Загрузка файла с изображением."""
    return st.file_uploader("Или загрузите изображение из файла")


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
            result = image_classification(loaded_image, processor, model)
        translated_result = translate_text(result, "en", "ru")
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
