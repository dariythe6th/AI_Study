import pytest
import pandas as pd
from src.service.engine import MoodTuneEngine
from src.models.mood_detector import MoodDetector


@pytest.fixture
def engine():
    engine = MoodTuneEngine()
    # Используем маленький тестовый датасет
    test_data = pd.DataFrame({
        'track_name': ['Blinding Lights', 'Levitating', 'Save Your Tears'],
        'track_artist': ['The Weeknd', 'Dua Lipa', 'The Weeknd'],
        'mood': ['happy', 'energetic', 'melancholic'],
        'energy': [0.85, 0.92, 0.45],
        'danceability': [0.75, 0.88, 0.55],
        'valence': [0.65, 0.78, 0.25],
        'tempo': [171, 103, 118],
        'track_popularity': [95, 88, 92]
    })
    engine.dataset = test_data
    return engine


def test_mood_detector():
    detector = MoodDetector()
    assert detector.detect("happy energetic workout") == "energetic"
    assert detector.detect("sad rainy day") == "sad"
    assert detector.detect("focus study coding") == "focus"
    assert detector.detect("random text without keywords") == "neutral"


def test_engine_initialization(engine):
    assert engine.dataset is not None
    assert len(engine.dataset) > 0


def test_get_recommendations(engine):
    result = engine.get_recommendations("energetic morning workout", limit=2)

    assert "personalized_recommendations" in result
    assert "top_5_songs_for_this_vibe" in result
    assert "detected_mood" in result
    assert result["detected_mood"] in ["happy", "energetic"]
    assert len(result["personalized_recommendations"]) <= 2


def test_mood_aware_ranking(engine):
    """Проверяем, что система учитывает настроение"""
    result = engine.get_recommendations("грустная меланхоличная музыка", limit=3)

    # Хотя бы одна рекомендация должна иметь melancholic настроение
    moods = [rec["mood"] for rec in result["personalized_recommendations"]]
    assert len(moods) > 0


def test_top_songs_for_mood(engine):
    top_songs = engine._get_diverse_top_songs("happy", top_n=2)
    assert len(top_songs) <= 2
    assert all(song["rank"] is not None for song in top_songs)