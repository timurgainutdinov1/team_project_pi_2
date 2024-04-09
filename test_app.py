"""test_app.py"""
from streamlit.testing.v1 import AppTest
import time

at = AppTest.from_file("image_classification_streamlit.py", default_timeout=1000).run()

def test_incorrect_url():
    at.text_input[0].set_value("https://www.google.com/").run()
    at.button[0].click().run()
    assert at.error[0].value == 'Не удалось найти изображение по указанной ссылке. Попробуйте снова!'

def test_null_url():
    at.text_input[0].set_value("").run()
    at.button[0].click().run()
    assert at.error[0].value == 'Не удалось найти изображение по указанной ссылке. Попробуйте снова!'

def test_correct_url():
    at.text_input[0].set_value("https://www.rgo.ru/sites/default/files/styles/head_image_article/public/node/61549/photo-2023-11-08-150058.jpeg").run()
    at.button[0].click().run()
    time.sleep(5)
    assert at.markdown[0].value == 'Результаты распознавания: :rainbow[tabby, tabby cat]'

