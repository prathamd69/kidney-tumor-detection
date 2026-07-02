import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def generate_dummy_image():
    """Generates a tiny 224x224 red square in memory to trick the API into thinking it's a real CT scan."""
    file = io.BytesIO()
    image = Image.new('RGB', (224, 224), color='red')
    image.save(file, 'jpeg')
    file.name = 'test.jpg'
    file.seek(0)
    return file

def test_home_endpoint():
    """Checks if the frontend HTML page loads properly."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_binary_prediction():
    """Sends a fake image to the binary endpoint and verifies the response format."""
    dummy_image = generate_dummy_image()
    response = client.post(
        "/predict/binary", 
        files={"file": ("test.jpg", dummy_image, "image/jpeg")}
    )

    assert response.status_code == 200

    json_data = response.json()
    assert "binary_screening" in json_data
    assert json_data["binary_screening"]["status"] == "success"
    assert "confidence" in json_data["binary_screening"]
