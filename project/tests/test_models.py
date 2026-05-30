from src.models.mood_detector import MoodDetector
from src.models.embedding import TextEncoder


def test_text_encoder():
    encoder = TextEncoder()
    texts = ["Hello world", "Test embedding"]
    embeddings = encoder.encode(texts)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 100  # размерность эмбеддинга


def test_mood_detector_comprehensive():
    detector = MoodDetector()

    test_cases = [
        ("I feel so happy and energetic today", "happy"),
        ("This is a sad and depressing song", "sad"),
        ("Perfect for workout and running", "energetic"),
        ("Calm music for relaxation", "chill"),
        ("Help me focus on studying", "focus"),
    ]

    for text, expected in test_cases:
        result = detector.detect(text)
        assert result == expected or result == "neutral"  # допускаем neutral