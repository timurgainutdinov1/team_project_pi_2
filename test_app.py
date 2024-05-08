"""Тесты для проверки приложения Streamlit."""

import time

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("image_classification_streamlit.py",
                       default_timeout=1000).run()


def test_incorrect_url():
    """Проверка ввода URL-адреса на объект,
    который не является изображением."""
    at.text_input[0].set_value("https://www.google.com/").run()
    at.button[0].click().run()
    assert at.error[0].value == (
        "Не удалось найти изображение по указанной ссылке. "
        "Попробуйте снова!"
    )


def test_null_url():
    """Проверка ввода пустого URL-адреса."""
    at.text_input[0].set_value("").run()
    at.button[0].click().run()
    assert at.error[0].value == (
        "Не удалось найти изображение по указанной ссылке. "
        "Попробуйте снова!"
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
    time.sleep(5)
    assert at.markdown[0].value == (
        "Результаты распознавания: "
        ":rainbow[tabby, tabby cat]"
    )
