from app.config import Settings
from app.rag import RAGService


if __name__ == "__main__":
    service = RAGService.build(Settings())
    print(f"Ingested {service.document_count} document(s) into {service.store.size} chunk(s).")
    print(f"Embeddings: {service.embedder.name}; LLM: {service.answerer.name}")
