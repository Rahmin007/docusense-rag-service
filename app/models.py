from typing import List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class SourceChunk(BaseModel):
    chunk_id: str
    similarity_score: float
    text_snippet: str
    source_file: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    tokens_used: int


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    embedding_provider: str
    llm_provider: str
