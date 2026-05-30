import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_and_clean_data(csv_path: str = "data/high_popularity_spotify_data.csv") -> pd.DataFrame:
    """Загрузка и базовая очистка данных"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # Базовая очистка
    df = df.drop_duplicates(subset=['track_name', 'track_artist'], keep='first')
    df['track_name'] = df['track_name'].fillna('Unknown').astype(str)
    df['track_artist'] = df['track_artist'].fillna('Unknown').astype(str)

    logger.info(f"Cleaned dataset shape: {df.shape}")
    return df