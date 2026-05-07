from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    section: str
    chunk: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: str  # "high" | "medium" | "low"
    fallback_reason: Optional[str] = None
    trace_id: str
    latency_ms: int
