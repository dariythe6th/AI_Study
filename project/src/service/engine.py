import logging
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any

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

    def initialize(self, csv_path: str = "data/high_popularity_spotify_data.csv"):
        if self.dataset is not None:
            return
        df = load_and_clean_data(csv_path)
        # Добавляем колонку 'mood' на основе аудио-фич
        df['mood'] = df.apply(self._infer_mood_from_features, axis=1)
        self.dataset = df
        self._index_tracks(df)
        logger.info(f"Engine ready with {len(df)} tracks")

    def _infer_mood_from_features(self, row: pd.Series) -> str:
        """Определяет настроение на основе energy, valence, danceability, tempo"""
        energy = float(row.get('energy', 0.5))
        valence = float(row.get('valence', 0.5))
        dance = float(row.get('danceability', 0.5))
        tempo = float(row.get('tempo', 120))

        if energy > 0.8 and valence > 0.8 and dance > 0.7:
            return 'ecstatic'
        if energy > 0.7 and valence > 0.7:
            return 'happy'
        if energy > 0.8 and tempo > 120:
            return 'energetic'
        if energy < 0.3 and valence < 0.3:
            return 'depressed'
        if energy < 0.4 and valence < 0.5:
            return 'melancholic'
        if energy < 0.5 and tempo < 100:
            return 'chill'
        if dance > 0.7 and energy > 0.6:
            return 'dance'
        if valence > 0.7 and energy < 0.6:
            return 'romantic'
        if energy > 0.6 and valence < 0.4:
            return 'angry'
        return 'neutral'

    def _index_tracks(self, df: pd.DataFrame):
        """Создаёт богатые текстовые описания треков для эмбеддингов"""
        track_texts = []
        ids = []
        metadatas = []

        for idx, row in df.iterrows():
            mood = row['mood']
            energy = float(row.get('energy', 0.5))
            dance = float(row.get('danceability', 0.5))
            valence = float(row.get('valence', 0.5))
            tempo = float(row.get('tempo', 120))

            text_parts = [
                f"Song: {row['track_name']}",
                f"Artist: {row['track_artist']}",
                f"Mood: {mood}",
                f"Energy: {'high' if energy > 0.7 else 'medium' if energy > 0.4 else 'low'}",
                f"Danceability: {'high' if dance > 0.7 else 'medium' if dance > 0.4 else 'low'}",
                f"Valence: {'positive' if valence > 0.6 else 'neutral' if valence > 0.4 else 'negative'}",
                f"Tempo: {'fast' if tempo > 120 else 'moderate' if tempo > 90 else 'slow'}",
                f"Genre: {row.get('playlist_genre', 'Various')}"
            ]
            vibe_text = " ".join(text_parts)

            track_id = str(row.get('track_id', f"track_{idx}"))
            metadata = {
                "song_name": row['track_name'],
                "artist_name": row['track_artist'],
                "album": row.get('track_album_name', 'Unknown'),
                "mood": mood,
                "popularity": int(row.get('track_popularity', 50)),
                "energy": energy,
                "danceability": dance,
                "valence": valence,
                "tempo": tempo,
                "genre": row.get('playlist_genre', 'Various')
            }
            track_texts.append(vibe_text)
            ids.append(track_id)
            metadatas.append(metadata)

        embeddings = self.encoder.encode(track_texts)
        self.vector_store.add_tracks(ids, embeddings, metadatas)
        logger.info(f"Indexed {len(track_texts)} tracks with rich mood description")

    def get_recommendations(self, query: str, limit: int = 12) -> Dict[str, Any]:
        detected_mood = self.mood_detector.detect(query)
        logger.info(f"Query: '{query}' → Mood: {detected_mood}")

        query_embedding = self.encoder.encode([query])[0]
        raw_results = self.vector_store.search(query_embedding, n_results=limit * 3)

        if not raw_results:
            return {"personalized_recommendations": [], "top_5_songs_for_this_vibe": [], "detected_mood": detected_mood}

        # Разнообразие: не более одного трека на артиста
        recommendations = self._diversify_recommendations(raw_results, detected_mood, limit)

        # Топ песен для этого настроения (разные артисты)
        top_songs = self._get_diverse_top_songs(detected_mood, 5)

        return {
            "personalized_recommendations": recommendations,
            "top_5_songs_for_this_vibe": top_songs,
            "detected_mood": detected_mood
        }

    def _diversify_recommendations(self, raw_results: List[Dict], detected_mood: str, limit: int) -> List[Dict]:
        """Группирует по артистам, берёт лучший трек от каждого"""
        artist_map = {}
        for res in raw_results:
            artist = res['metadata']['artist_name']
            if artist not in artist_map:
                artist_map[artist] = []
            artist_map[artist].append(res)

        candidates = []
        for artist, tracks in artist_map.items():
            best = max(tracks, key=lambda x: x['similarity'])  # similarity, а не score
            candidates.append(best)

        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        final = candidates[:limit]

        recommendations = []
        for res in final:
            meta = res['metadata']
            similarity = res['similarity']
            match_pct = min(int(similarity * 100), 100)  # теперь правильно
            why = self._generate_why_it_matches(meta, detected_mood, match_pct)
            recommendations.append({
                "song_name": meta['song_name'],
                "artist_name": meta['artist_name'],
                "album": meta['album'],
                "mood": meta['mood'],
                "popularity": meta['popularity'],
                "match_percentage": match_pct,
                "energy": meta.get('energy'),
                "danceability": meta.get('danceability'),
                "why_it_matches": why
            })
        return recommendations

    def _generate_why_it_matches(self, meta: dict, detected_mood: str, match_pct: int) -> str:
        phrases = [
            f"Идеально передаёт атмосферу вашего запроса",
            f"Отлично подходит под настроение «{detected_mood}»",
            f"Этот трек очень точно ловит вайб, который вы описали",
            f"Мощное попадание в настроение",
            f"Классика для такого состояния"
        ]
        if meta.get('mood') == detected_mood and match_pct > 75:
            phrases.append(f"Прямо в точку — идеальное попадание в {detected_mood} настроение")
        if meta.get('energy', 0) > 0.75:
            phrases.append("Заряжает энергией и поднимает настроение")
        return random.choice(phrases)

    def _get_diverse_top_songs(self, mood: str, top_n: int = 5) -> List[Dict]:
        if self.dataset is None or 'mood' not in self.dataset.columns:
            return []
        subset = self.dataset[self.dataset['mood'] == mood]
        if subset.empty:
            return []
        # Убираем дубликаты по артистам
        subset = subset.drop_duplicates(subset=['track_artist'])
        top = subset.nlargest(top_n, 'track_popularity')
        return [
            {
                "rank": i + 1,
                "song_name": row['track_name'],
                "artist_name": row['track_artist'],
                "album": row.get('track_album_name', 'Unknown'),
                "popularity": int(row.get('track_popularity', 0))
            }
            for i, row in top.iterrows()
        ]