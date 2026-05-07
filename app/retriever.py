import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.indexer import Chunk


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([c.tokens for c in chunks])
        self._embeddings: Optional[np.ndarray] = None

    def set_embeddings(self, embeddings: list[list[float]]) -> None:
        self._embeddings = np.array(embeddings, dtype=np.float32)

    def search(
        self,
        query: str,
        query_embedding: Optional[list[float]],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        query_tokens = re.findall(r"\w+", query.lower())
        bm25_raw = self.bm25.get_scores(query_tokens)

        bm25_max = bm25_raw.max()
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        if self._embeddings is not None and query_embedding is not None:
            q_vec = np.array(query_embedding, dtype=np.float32)
            sem_scores = self._embeddings @ q_vec
        else:
            sem_scores = np.zeros(len(self.chunks), dtype=np.float32)

        n = len(self.chunks)
        k = 60
        bm25_ranks = n - bm25_norm.argsort().argsort()
        sem_ranks = n - sem_scores.argsort().argsort()
        rrf = 1.0 / (k + bm25_ranks) + 1.0 / (k + sem_ranks)

        top_idx = rrf.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(rrf[i])) for i in top_idx]
