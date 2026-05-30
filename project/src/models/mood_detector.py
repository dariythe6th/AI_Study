import re
from typing import Dict

class MoodDetector:
    def __init__(self):
        self.mood_keywords: Dict[str, list] = {
            'happy': ['happy', 'joy', 'glad', 'excited', 'great'],
            'sad': ['sad', 'lonely', 'depressed', 'cry'],
            'energetic': ['energetic', 'workout', 'energy', 'hyped'],
            'chill': ['chill', 'relax', 'calm', 'peaceful'],
            'focus': ['focus', 'study', 'work', 'coding'],
            'romantic': ['romantic', 'love', 'date'],
            'party': ['party', 'dance', 'club'],
        }

    def detect(self, text: str) -> str:
        text = text.lower()
        scores = {mood: 0 for mood in self.mood_keywords}

        for mood, words in self.mood_keywords.items():
            for word in words:
                if re.search(r'\b' + re.escape(word) + r'\b', text):
                    scores[mood] += 1

        best_mood = max(scores.items(), key=lambda x: x[1])
        return best_mood[0] if best_mood[1] > 0 else 'neutral'