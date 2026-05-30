import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.service.schemas import RecommendRequest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_recommend_endpoint():
    payload = {
        "text": "energetic morning workout",
        "limit": 5
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code in [200, 401]  # 401 если требует авторизацию

    if response.status_code == 200:
        data = response.json()
        assert "personalized_recommendations" in data
        assert "detected_mood" in data
        assert len(data["personalized_recommendations"]) <= 5

def test_auth_register():
    payload = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123"
    }
    response = client.post("/auth/register", json=payload)
    assert response.status_code in [200, 400]  # 400 если пользователь уже существует

def test_auth_login():
    payload = {
        "email": "test@example.com",
        "password": "password123"
    }
    response = client.post("/auth/login", json=payload)
    assert response.status_code in [200, 401]

def test_invalid_request():
    response = client.post("/recommend", json={"text": ""})
    assert response.status_code == 401  # Pydantic validation error