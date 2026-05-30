import logging
import numpy as np
from typing import List, Dict, Any
import pandas as pd

from src.data.loader import load_and_clean_data
from src.models.embedding import TextEncoder
from src.models.vector_store import VectorStore
from src.models.mood_detector import MoodDetector
from src.config import settings

logger = logging.getLogger(__name__)

class MoodTuneEngine:
    def __init__(self):
        self.encoder = TextEncoder(settings.MODEL_NAME)
        self.vector_store = VectorStore(settings.CHROMA_PERSIST_DIRECTORY)
        self.mood_detector = MoodDetector()
        self.dataset: pd.DataFrame | None = None
        logger.info("MoodTuneEngine initialized — focus on mood & vibe")

    def initialize(self, csv_path: str = "data/high_popularity_spotify_data.csv"):
        if self.dataset is not None:
            return

        df = load_and_clean_data(csv_path)
        self.dataset = df
        self._index_with_mood_understanding(df)
        logger.info(f"Engine ready with {len(df)} tracks — mood-aware indexing")

    def _index_with_mood_understanding(self, df: pd.DataFrame):
        """Создаём эмбеддинги с сильным акцентом на настроение и атмосферу"""
        track_texts = []
        ids = []
        metadatas = []

        for idx, row in df.iterrows():
            mood = row.get('mood', 'neutral')
            energy = float(row.get('energy', 0.5))
            dance = float(row.get('danceability', 0.5))
            valence = float(row.get('valence', 0.5))
            tempo = float(row.get('tempo', 120))

            # === Ключевое улучшение: очень описательный текст ===
            vibe_text = f"""
            Song title: {row.get('track_name', '')}
            Artist: {row.get('track_artist', '')}
            Overall mood: {mood}
            Emotional atmosphere: {self._describe_emotion(energy, valence)}
            Energy feel: {self._describe_energy(energy)}
            Movement quality: {self._describe_dance(dance)}
            Tempo character: {self._describe_tempo(tempo)}
            Genre vibe: {row.get('playlist_genre', 'Various')}
            Suggested for: {self._suggest_context(mood, energy, valence)}
            """

            track_id = str(row.get('track_id', f"track_{idx}"))

            metadata = {
                "song_name": row.get('track_name', 'Unknown'),
                "artist_name": row.get('track_artist', 'Unknown'),
                "album": row.get('track_album_name', 'Unknown'),
                "mood": mood,
                "popularity": int(row.get('track_popularity', 50)),
                "energy": energy,
                "danceability": dance,
                "valence": valence,
                "tempo": tempo,
                "genre": row.get('playlist_genre', 'Various')
            }

            track_texts.append(vibe_text.strip())
            ids.append(track_id)
            metadatas.append(metadata)

        embeddings = self.encoder.encode(track_texts)
        self.vector_store.add_tracks(ids, embeddings, metadatas)
        logger.info(f"Indexed {len(track_texts)} tracks with deep mood understanding")

    # ==================== ВСПОМОГАТЕЛЬНЫЕ ОПИСАНИЯ ====================
    def _describe_emotion(self, energy, valence):
        if valence > 0.65 and energy > 0.6: return "joyful uplifting"
        if valence < 0.4 and energy < 0.5: return "melancholic introspective"
        if valence > 0.7: return "positive bright"
        if valence < 0.35: return "sad emotional"
        return "complex nuanced"

    def _describe_energy(self, energy):
        if energy > 0.8: return "high powered explosive"
        if energy > 0.65: return "energetic driving"
        if energy > 0.45: return "moderate balanced"
        return "calm relaxed"

    def _describe_dance(self, dance):
        if dance > 0.8: return "very danceable groovy"
        if dance > 0.65: return "rhythmic danceable"
        if dance > 0.45: return "light movement"
        return "stationary calm"

    def _describe_tempo(self, tempo):
        if tempo > 140: return "fast upbeat"
        if tempo > 110: return "mid-tempo flowing"
        if tempo > 80: return "moderate"
        return "slow atmospheric"

    def _suggest_context(self, mood, energy, valence):
        if mood == "happy" or valence > 0.7:
            return "celebration, party, good mood, sunny day"
        if mood == "sad" or valence < 0.35:
            return "rainy day, reflection, emotional moments, night drive"
        if energy > 0.75:
            return "workout, motivation, energetic activity"
        if energy < 0.4:
            return "relaxation, study, sleep, deep focus"
        return "general listening, background, various situations"

    # ==================== РЕКОМЕНДАЦИИ ====================
    def get_recommendations(self, query: str, limit: int = 12) -> Dict[str, Any]:
        detected_mood = self.mood_detector.detect(query)
        logger.info(f"Query: '{query}' → Mood: {detected_mood}")

        query_embedding = self.encoder.encode([query])[0]

        # Ищем много кандидатов
        results = self.vector_store.search(query_embedding, n_results=limit * 6)

        recommendations = self._mood_aware_reranking(results, limit, detected_mood, query)

        top_songs = self._get_top_songs_for_mood(detected_mood, 5)

        return {
            "personalized_recommendations": recommendations,
            "top_5_songs_for_this_vibe": top_songs,
            "detected_mood": detected_mood
        }

    def _mood_aware_reranking(self, results: Dict, limit: int, detected_mood: str, query: str) -> List[Dict]:
        if not results.get('ids') or not results['ids'][0]:
            return []

        candidates = []
        seen = set()

        distances = results['distances'][0]
        max_dist = max(distances) if distances else 1.0
        min_dist = min(distances) if distances else 0.0

        for i in range(len(results['ids'][0])):
            if len(candidates) >= limit * 3:
                break

            meta = results['metadatas'][0][i]
            artist = meta['artist_name']
            if artist in seen:
                continue
            seen.add(artist)

            raw_distance = distances[i]

            if max_dist - min_dist > 0.001:
                normalized_score = (max_dist - raw_distance) / (max_dist - min_dist)
            else:
                normalized_score = 1.0 - raw_distance

            mood_bonus = 0.45 if meta.get('mood') == detected_mood else 0.0
            popularity_bonus = meta.get('popularity', 50) / 300.0

            final_score = (
                    normalized_score * 0.55 +
                    mood_bonus * 0.35 +
                    popularity_bonus * 0.10
            )

            match_percentage = min(int(final_score * 100), 100)

            # === Красивые объяснения ===
            why = self._generate_why_it_matches(
                meta, detected_mood, query, match_percentage
            )

            candidates.append({
                "song_name": meta['song_name'],
                "artist_name": meta['artist_name'],
                "album": meta['album'],
                "mood": meta['mood'],
                "popularity": meta['popularity'],
                "match_percentage": match_percentage,
                "energy": meta['energy'],
                "danceability": meta['danceability'],
                "valence": meta['valence'],
                "tempo": meta['tempo'],
                "why_it_matches": why
            })

        candidates.sort(key=lambda x: (x['match_percentage'], x['popularity']), reverse=True)
        return candidates[:limit]

    def _generate_why_it_matches(self, meta: dict, detected_mood: str, query: str, percentage: int) -> str:
        """Генерирует красивые естественные объяснения"""
        song = meta['song_name']
        artist = meta['artist_name']
        mood = meta.get('mood', 'neutral')
        energy = meta.get('energy', 0.5)

        phrases = [
            f"Идеально передаёт атмосферу вашего запроса",
            f"Отлично подходит под настроение «{query}»",
            f"Этот трек очень точно ловит вайб, который вы описали",
            f"Мощное попадание в настроение",
            f"Одна из лучших песен для такого состояния",
            f"Классика для такого настроения",
            f"Почти идеально соответствует вашему описанию",
            f"Эмоционально очень близко к тому, что вы ищете",
            f"Этот трек словно создан для вашего запроса",
            f"Сильное совпадение по энергетике и эмоциям",
        ]

        # Дополнительная персонализация
        if mood == detected_mood and percentage > 75:
            phrases.extend([
                f"Прямо в точку — идеальное попадание в {mood} настроение",
                f"Один из лучших треков для {detected_mood} состояния",
            ])

        if energy > 0.75:
            phrases.append("Заряжает энергией и поднимает настроение")

        import random
        return random.choice(phrases)

    def _get_top_songs_for_mood(self, mood: str, top_n: int = 5) -> List[Dict]:
        if self.dataset is None or 'mood' not in self.dataset.columns:
            return []
        subset = self.dataset[self.dataset['mood'] == mood]
        if subset.empty:
            return []
        top = subset.nlargest(top_n, 'track_popularity')
        return [
            {
                "rank": i + 1,
                "song_name": row['track_name'],
                "artist_name": row['track_artist'],
                "album": row.get('track_album_name'),
                "popularity": int(row.get('track_popularity', 0))
            }
            for i, row in top.iterrows()
        ]