import logging

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from .config import Settings
from .models import IngestResponse, QueryRequest, QueryResponse
from .rag import RAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("docusense.api")

app = FastAPI(
    title="DocuSense",
    version="1.0.0",
    description="Lightweight grounded RAG service with deterministic fallback guardrails.",
)

settings = Settings()
service = RAGService.build(settings)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # rag.py already turns answer-generation failures into a graceful
    # fallback response; this is the safety net for anything upstream of
    # that (e.g. the embedding provider itself erroring or timing out) so
    # a provider outage returns a clean 500 instead of leaking a traceback.
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": service.document_count,
        "chunks": service.store.size,
        "embedding_provider": service.embedder.name,
        "llm_provider": service.answerer.name,
    }


@app.post("/api/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    service.ingest()
    return IngestResponse(
        documents=service.document_count,
        chunks=service.store.size,
        embedding_provider=service.embedder.name,
        llm_provider=service.answerer.name,
    )


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    return service.query(payload.question)
