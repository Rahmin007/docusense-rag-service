from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import os
from typing import Iterable

import numpy as np


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbeddingProvider(EmbeddingProvider):
    """Offline deterministic vectorizer used for local demos and tests."""

    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=12000,
        )
        self._fitted = False

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray().astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TF-IDF provider must be fit before querying")
        return self._vectorizer.transform([text]).toarray()[0].astype(np.float32)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, model: str, api_key: str | None) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI embedding provider")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return np.asarray([item.embedding for item in ordered], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return np.asarray(response.data[0].embedding, dtype=np.float32)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    name = "sentence_transformer"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)


def build_embedding_provider(provider: str, *, model: str, sentence_model: str, api_key: str | None) -> EmbeddingProvider:
    provider = provider.lower().strip()
    if provider == "tfidf":
        return TfidfEmbeddingProvider()
    if provider == "openai":
        return OpenAIEmbeddingProvider(model=model, api_key=api_key)
    if provider in {"sentence_transformer", "sentence-transformer", "hf"}:
        return SentenceTransformerEmbeddingProvider(sentence_model)
    raise ValueError(f"Unknown embedding provider: {provider}")
