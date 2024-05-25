"""Тесты для проверки приложения Streamlit."""

import time
import pytest
import io
from PIL import Image, UnidentifiedImageError
from streamlit.testing.v1 import AppTest
from model import image_classification, load_processor, load_model

# Создаем объект AppTest для тестирования приложения Streamlit
at = AppTest.from_file("main.py", default_timeout=1000).run()


@pytest.fixture(scope="module")
def processor_and_model():
    """Фикстура для загрузки процессора и модели."""
    processor = load_processor()
    model = load_model()
    return processor, model


def test_no_image_url():
    """Проверка ввода URL-адреса на объект, который не является изображением"""
    at.text_input[0].set_value("https://www.google.com/").run()
    at.button[0].click().run()
    assert at.error[0].value == (
        "Ваша ссылка или файл не содержат изображения. "
        "Предоставьте корректную ссылку или файл "
        "и попробуйте снова!"
    )


def test_null_url():
    """Проверка ввода пустого URL-адреса."""
    at.text_input[0].set_value("").run()
    at.button[0].click().run()
    assert at.error[0].value == (
        "Вы не предоставили источник "
        "для загрузки изображения. "
        "Загрузите файл с изображением или укажите ссылку "
        "и попробуйте снова!"
    )


def test_correct_url():
    """Проверка ввода корректного URL-адреса на изображение."""
    (
        at.text_input[0]
        .set_value(
            "https://www.rgo.ru/sites/default/files/"
            "styles/head_image_article/public/node/"
            "61549/photo-2023-11-08-150058.jpeg"
        )
        .run()
    )
    at.button[0].click().run()
    time.sleep(5)  # Добавляем ожидание 5 секунд
    assert at.markdown[0].value == (
        "Результаты распознавания: табби, полосатый кот"
    )


def test_incorrect_url():
    """Проверка ввода некорректного URL-адреса"""
    at.text_input[0].set_value("1234").run()
    at.button[0].click().run()
    assert at.error[0].value == (
        "Некорректная ссылка! "
        "Укажите корректную ссылку "
        "и попробуйте снова!"
    )


def test_correct_image_file(processor_and_model):
    """Проверка загрузки изображения через файл."""
    processor, model = processor_and_model
    with open("test_image.jpg", "rb") as file:
        test_image_bytes = file.read()
    test_image = Image.open(io.BytesIO(test_image_bytes))
    try:
        result = image_classification(test_image, processor, model)
        assert result == "Egyptian cat"
    except UnidentifiedImageError:
        assert False, "Ошибка при обработке изображения"
