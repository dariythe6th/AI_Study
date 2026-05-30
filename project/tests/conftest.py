import pytest
import pandas as pd

@pytest.fixture(scope="session")
def sample_dataset():
    return pd.DataFrame({
        'track_name': ['Song A', 'Song B', 'Song C'],
        'track_artist': ['Artist 1', 'Artist 2', 'Artist 1'],
        'mood': ['happy', 'sad', 'happy'],
        'energy': [0.8, 0.3, 0.9],
        'danceability': [0.7, 0.4, 0.85],
        'valence': [0.75, 0.2, 0.8],
        'tempo': [120, 80, 140]
    })