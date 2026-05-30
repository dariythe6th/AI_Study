import chromadb
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "moodtune_tracks"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.local_vectors = []  # fallback

        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info("Connected to ChromaDB")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}. Using local storage.")
            self.client = None

    def add_tracks(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict]):
        if self.collection is not None:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
        else:
            for idx, emb, meta in zip(ids, embeddings, metadatas):
                self.local_vectors.append((idx, emb, meta))
        logger.info(f"Added/updated {len(ids)} tracks")

    def search(self, query_embedding: np.ndarray, n_results: int = 20) -> List[Dict]:
        if self.collection is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            hits = []
            if results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    hits.append({
                        'id': results['ids'][0][i],
                        'score': results['distances'][0][i],
                        'metadata': results['metadatas'][0][i] if results.get('metadatas') else {}
                    })
            return hits
        else:
            if not self.local_vectors:
                return []
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
            sims = []
            for vid, vec, meta in self.local_vectors:
                v_norm = vec / (np.linalg.norm(vec) + 1e-9)
                sims.append((vid, float(np.dot(v_norm, q_norm)), meta))
            sims.sort(key=lambda x: x[1], reverse=True)
            return [{'id': s[0], 'score': s[1], 'metadata': s[2]} for s in sims[:n_results]]