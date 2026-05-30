import pytest
import numpy as np
from src.models.vector_store import VectorStore

@pytest.fixture
def vector_store():
    store = VectorStore(persist_directory="./test_chroma")
    return store

def test_vector_store_add_and_search(vector_store):
    ids = ["track1", "track2"]
    embeddings = np.random.rand(2, 384).astype(np.float32)  # 384 — размерность MiniLM
    metadatas = [
        {"song_name": "Test Song 1", "artist_name": "Artist A"},
        {"song_name": "Test Song 2", "artist_name": "Artist B"}
    ]

    vector_store.add_tracks(ids, embeddings, metadatas)

    query = np.random.rand(384).astype(np.float32)
    results = vector_store.search(query, n_results=2)

    assert len(results) > 0
    assert 'ids' in results or isinstance(results, list)


def test_vector_store_empty_search(vector_store):
    query = np.zeros(384, dtype=np.float32)
    results = vector_store.search(query, n_results=5)
    # Не должно падать
    assert isinstance(results, dict) or isinstance(results, list)