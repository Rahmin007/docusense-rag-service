from dataclasses import dataclass
import numpy as np

from .chunking import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    similarity_score: float


class InMemoryVectorStore:
    """Small embedded vector store using normalized numpy cosine similarity."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count must match")
        if not chunks:
            self._chunks = []
            self._vectors = np.empty((0, 0), dtype=np.float32)
            return
        self._chunks = list(chunks)
        self._vectors = self._normalize(np.asarray(embeddings, dtype=np.float32))

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query = self._normalize(query)[0]
        scores = self._vectors @ query
        order = np.argsort(-scores)[: max(1, top_k)]
        return [RetrievedChunk(self._chunks[i], float(scores[i])) for i in order]

    @property
    def size(self) -> int:
        return len(self._chunks)
