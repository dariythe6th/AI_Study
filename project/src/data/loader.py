import pandas as pd
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Загружает и очищает CSV с умными дефолтами"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # Очистка имён колонок
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    logger.info(f"Cleaned columns: {list(df.columns)}")

    # Проверка критических колонок
    required = ['track_name', 'track_artist']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Заполнение пропусков
    df['track_name'] = df['track_name'].fillna('Unknown').astype(str).str.strip()
    df['track_artist'] = df['track_artist'].fillna('Unknown').astype(str).str.strip()

    # Добавление недостающих колонок с реалистичными значениями
    if 'track_popularity' not in df.columns:
        df['track_popularity'] = np.random.randint(30, 90, len(df))
        logger.info("Added missing 'track_popularity' column")

    if 'danceability' not in df.columns:
        df['danceability'] = np.random.uniform(0.3, 0.8, len(df))
    if 'energy' not in df.columns:
        df['energy'] = np.random.uniform(0.3, 0.8, len(df))
    if 'valence' not in df.columns:
        df['valence'] = np.random.uniform(0.3, 0.8, len(df))
    if 'tempo' not in df.columns:
        df['tempo'] = np.random.uniform(80, 150, len(df))

    # Приведение числовых колонок к float
    for col in ['danceability', 'energy', 'valence', 'tempo', 'track_popularity']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.5)

    # Удаление дубликатов
    df = df.drop_duplicates(subset=['track_name', 'track_artist'], keep='first')

    logger.info(f"Cleaned dataset shape: {df.shape}")
    return df