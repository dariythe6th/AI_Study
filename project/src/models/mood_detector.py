import re
from typing import Dict

class MoodDetector:
    def __init__(self):
        self.mood_keywords: Dict[str, list] = {
            'happy': ['happy', 'joy', 'glad', 'excited', 'love', 'great', 'good', 'amazing', 'wonderful', 'sunny'],
            'sad': ['sad', 'lonely', 'depressed', 'cry', 'heartbroken', 'miserable', 'tears', 'broken', 'rainy'],
            'energetic': ['energetic', 'powerful', 'motivated', 'pumped', 'hyped', 'workout', 'exercise', 'energy', 'fast'],
            'chill': ['relax', 'calm', 'peaceful', 'chill', 'sleepy', 'lofi', 'quiet', 'meditation', 'cozy'],
            'focus': ['study', 'focus', 'concentrate', 'work', 'productive', 'thinking', 'coding', 'reading'],
            'romantic': ['romantic', 'love', 'heart', 'kiss', 'beautiful', 'sweet', 'relationship', 'date'],
            'party': ['party', 'dance', 'club', 'festival', 'celebration', 'night', 'fun', 'friends'],
            'angry': ['angry', 'mad', 'frustrated', 'rage', 'hate', 'annoyed', 'pissed'],
            'melancholic': ['melancholy', 'nostalgic', 'bittersweet', 'thoughtful', 'reflective']
        }

    def detect(self, text: str) -> str:
        text = text.lower()
        scores = {mood: 0 for mood in self.mood_keywords}
        for mood, words in self.mood_keywords.items():
            for word in words:
                if re.search(r'\b' + re.escape(word) + r'\b', text):
                    scores[mood] += 1
        best_mood, best_score = max(scores.items(), key=lambda x: x[1])
        return best_mood if best_score > 0 else 'neutral'