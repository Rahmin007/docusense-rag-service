from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_documents
from .config import Settings
from .embeddings import EmbeddingProvider, build_embedding_provider
from .llm import AnswerProvider, FALLBACK_MESSAGE, build_answer_provider
from .models import QueryResponse, SourceChunk
from .vector_store import InMemoryVectorStore, RetrievedChunk


@dataclass
class RAGService:
    settings: Settings
    embedder: EmbeddingProvider
    answerer: AnswerProvider
    store: InMemoryVectorStore
    chunks: list
    document_count: int

    @classmethod
    def build(cls, settings: Settings) -> "RAGService":
        embedder = build_embedding_provider(
            settings.embedding_provider,
            model=settings.openai_embedding_model,
            sentence_model=settings.sentence_transformer_model,
            api_key=settings.openai_api_key,
        )
        answerer = build_answer_provider(
            settings.llm_provider,
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            anthropic_model=settings.anthropic_chat_model,
            anthropic_api_key=settings.anthropic_api_key,
        )
        service = cls(settings, embedder, answerer, InMemoryVectorStore(), [], 0)
        service.ingest()
        return service

    def ingest(self) -> None:
        documents = self._load_documents(self.settings.resolved_docs_dir)
        chunks = chunk_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise RuntimeError(f"No .md or .txt documents found in {self.settings.resolved_docs_dir}")
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        self.chunks = chunks
        self.document_count = len(documents)

    @staticmethod
    def _load_documents(docs_dir: Path) -> list[tuple[str, str]]:
        docs_dir.mkdir(parents=True, exist_ok=True)
        supported = []
        for path in sorted(docs_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                supported.append((path.relative_to(docs_dir).as_posix(), path.read_text(encoding="utf-8")))
        return supported

    @staticmethod
    def _snippet(text: str, question: str, max_chars: int) -> str:
        import re

        if len(text) <= max_chars:
            return text

        terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+", question)
            if len(token) > 2
        }
        blocks = [b.strip() for b in re.split(r"\n\n+", text) if b.strip()]
        if not blocks:
            return text[:max_chars] + "..."

        def block_score(block: str) -> int:
            words = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", block)}
            return sum(1 for term in terms if term in words or term.rstrip("s") in words)

        best = max(blocks, key=block_score)
        if len(best) > max_chars:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", best) if s.strip()]
            if sentences:
                best = max(sentences, key=block_score)
        if len(best) <= max_chars:
            return best

        # Long evidence sentences are centered around the first matching term.
        lower = best.lower()
        positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
        start = max(0, (min(positions) if positions else 0) - max_chars // 3)
        snippet = best[start : start + max_chars]
        if start > 0:
            snippet = "..." + snippet
        if start + max_chars < len(best):
            snippet += "..."
        return snippet

    def query(self, question: str) -> QueryResponse:
        query_embedding = self.embedder.embed_query(question)
        retrieved = self.store.search(query_embedding, self.settings.top_k)

        accepted = [r for r in retrieved if r.similarity_score >= self.settings.similarity_threshold]
        sources = [
            SourceChunk(
                chunk_id=item.chunk.chunk_id,
                similarity_score=round(item.similarity_score, 4),
                text_snippet=self._snippet(item.chunk.text, question, self.settings.snippet_chars),
                source_file=item.chunk.source_file,
            )
            for item in accepted
        ]

        if not accepted:
            return QueryResponse(answer=FALLBACK_MESSAGE, sources=[], tokens_used=0)

        # The offline extractive provider is intentionally conservative: most
        # of the question's meaningful terms must appear (as whole words) in the
        # retrieved evidence. A single incidental overlap such as "limit" vs
        # "limited" is not enough to count as support.
        # The OpenAI/Anthropic paths rely on the structured support/citation gate below.
        if self.answerer.name == "extractive":
            import re

            def normalize(word: str) -> str:
                word = word.lower()
                for suffix in ("ies", "ed", "es", "s"):
                    if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                        return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
                return word

            stopwords = {
                "what", "is", "the", "a", "an", "of", "on", "for", "to",
                "and", "or", "how", "long", "are", "does", "do", "in", "with",
                "policy", "period", "periods", "tell", "me", "please", "can",
                "who", "when", "where", "which", "why", "many", "much", "there",
                "our", "your", "any", "about",
            }
            q_terms = {
                normalize(token)
                for token in re.findall(r"[A-Za-z0-9]+", question)
                if len(token) > 2 and token.lower() not in stopwords
            }
            evidence_words = {
                normalize(token)
                for item in accepted
                for token in re.findall(r"[A-Za-z0-9]+", item.chunk.text)
            }
            lexical_hits = len(q_terms & evidence_words)
            if q_terms and lexical_hits / len(q_terms) < 0.5:
                return QueryResponse(answer=FALLBACK_MESSAGE, sources=sources, tokens_used=0)

        try:
            generated = self.answerer.answer(question, accepted)
        except Exception:
            return QueryResponse(answer=FALLBACK_MESSAGE, sources=sources, tokens_used=0)

        allowed_ids = {s.chunk_id for s in sources}
        citations = set(generated.cited_chunk_ids)
        supported = (
            bool(generated.answer)
            and generated.answer != FALLBACK_MESSAGE
            and bool(citations)
            and citations.issubset(allowed_ids)
        )
        if not supported:
            return QueryResponse(answer=FALLBACK_MESSAGE, sources=sources, tokens_used=generated.tokens_used)

        return QueryResponse(answer=generated.answer, sources=sources, tokens_used=generated.tokens_used)
