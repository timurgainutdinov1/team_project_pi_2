from streamlit.testing.v1 import AppTest

at = AppTest.from_file("image_classification_streamlit.py", default_timeout=1000).run()


def test_correct_url():
    """
    Пользователь вводит корректную ссылку на изображение
    """
    at.text_input[0].set_value("https://goo.su/TDcn").run()
    at.button[0].click().run()
    assert "Результаты распознавания: :rainbow[tabby, tabby cat]" in at.markdown[0].value


def test_incorrect_url():
    """
    Пользователь вводит некорректную ссылку на изображение
    (ссылку на объект не являющийся изображением)
    """
    at.text_input[0].set_value("https://www.google.com/").run()
    at.button[0].click().run()
    assert at.error[0].value == "Не удалось найти изображение по указанной ссылке. Попробуйте снова!"


def test_null_url():
    """
    Пользователь не вводит ссылку на изображение
    (оставляет поле для ввода ссылки пустым)
    """
    at.text_input[0].set_value("").run()
    at.button[0].click().run()
    assert at.error[0].value == "Не удалось найти изображение по указанной ссылке. Попробуйте снова!"
