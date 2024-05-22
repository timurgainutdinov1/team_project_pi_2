import pytest

from fastapi.testclient import TestClient
from image_classification_fastapi import app, load_image

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Image classification API"}

@pytest.fixture
def mock_image():
    """Фикстура для загрузки изображения."""
    from PIL import Image
    import io
    
    # Создание простого черного изображения для тестов
    image = Image.new('RGB', (224, 224), color='black')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

class MockResponse:
    def __init__(self, raw):
        self.raw = raw

def test_load_image(mocker, mock_image):
    """Тест для функции загрузки изображения."""
    mock_response = MockResponse(mock_image)
    mocker.patch('requests.get', return_value=mock_response)
    url = "https://www.rgo.ru/sites/default/files/styles/head_image_article/public/node/61549/photo-2023-11-08-150058.jpeg"
    image = load_image(url)
    assert image is not None