import chromadb
from chromadb.config import Settings
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="moodtune_tracks")

    def add_tracks(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict]):
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        logger.info(f"Added/updated {len(ids)} tracks in vector store")

    def search(self, query_embedding: np.ndarray, n_results: int = 20):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        return results