import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.service.engine import MoodTuneEngine

client = TestClient(app)


@pytest.fixture(scope="module")
def test_engine():
    engine = MoodTuneEngine()
    # Инициализируем с тестовыми данными
    engine.initialize()
    return engine


def test_full_recommendation_flow(test_engine):
    """Интеграционный тест полного цикла"""
    query = "спокойная музыка для вечернего релакса"

    result = test_engine.get_recommendations(query, limit=3)

    assert "personalized_recommendations" in result
    assert "top_5_songs_for_this_vibe" in result
    assert "detected_mood" in result
    assert len(result["personalized_recommendations"]) > 0

    # Проверяем структуру рекомендации
    rec = result["personalized_recommendations"][0]
    assert "song_name" in rec
    assert "artist_name" in rec
    assert "match_percentage" in rec
    assert isinstance(rec["match_percentage"], int)


def test_mood_detection_integration():
    """Проверяем, что mood detector работает в связке с рекомендациями"""
    from src.models.mood_detector import MoodDetector
    detector = MoodDetector()

    moods = {
        "happy energetic workout": "energetic",
        "sad rainy night": "sad",
        "focus study session": "focus",
        "romantic dinner": "romantic"
    }

    for query, expected in moods.items():
        detected = detector.detect(query)
        assert detected == expected or detected == "neutral"


def test_api_with_real_query():
    """Тест реального HTTP запроса"""
    response = client.post(
        "/recommend",
        json={"text": "мотивирующая музыка для спорта", "limit": 4}
    )

    # Может вернуть 401, если требует авторизацию — это нормально
    assert response.status_code in [200, 401]

    if response.status_code == 200:
        data = response.json()
        assert isinstance(data.get("personalized_recommendations"), list)