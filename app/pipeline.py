import asyncio
import os
from pathlib import Path
from typing import Optional

import httpx

from app.indexer import Chunk, load_and_index
from app.llm import generate_answer
from app.models import AskResponse, Source
from app.reranker import rerank
from app.retriever import HybridRetriever
from app.tracer import PipelineTrace, new_trace

FALLBACK_THRESHOLD = float(os.getenv("FALLBACK_SCORE_THRESHOLD", "0.2"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
TOP_N = int(os.getenv("RERANKER_TOP_N", "3"))

_retriever: Optional[HybridRetriever] = None
_http_client: Optional[httpx.AsyncClient] = None


def _get_retriever() -> HybridRetriever:
    if _retriever is None:
        raise RuntimeError("Pipeline not initialized — call init_pipeline() at startup")
    return _retriever


def _confidence(score: float, is_fallback: bool) -> str:
    if is_fallback:
        return "low"
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


async def _embed_passages(texts: list[str], api_key: str, client: httpx.AsyncClient) -> list[list[float]]:
    batch_size = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = await client.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "model": "jina-embeddings-v3",
                "input": batch,
                "task": "retrieval.passage",
                "dimensions": 1024,
                "normalized": True,
                "embedding_type": "float",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        all_embeddings.extend(d["embedding"] for d in sorted_data)
    return all_embeddings


async def _embed_query(query: str, api_key: str, client: httpx.AsyncClient) -> Optional[list[float]]:
    resp = await client.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "model": "jina-embeddings-v3",
            "input": [query],
            "task": "retrieval.query",
            "dimensions": 1024,
            "normalized": True,
            "embedding_type": "float",
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]


async def init_pipeline(kb_path: Path) -> int:
    global _retriever, _http_client

    _http_client = httpx.AsyncClient(
        timeout=60,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )

    chunks = load_and_index(kb_path)
    retriever = HybridRetriever(chunks)

    api_key = os.getenv("JINA_API_KEY", "")
    if api_key:
        embeddings = await _embed_passages([c.text for c in chunks], api_key, _http_client)
        retriever.set_embeddings(embeddings)

    _retriever = retriever
    return len(chunks)


async def close_pipeline() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


async def process_question(question: str) -> AskResponse:
    trace: PipelineTrace = new_trace(question)
    retriever = _get_retriever()
    client = _http_client

    # Stage 1: context loading
    trace.record_stage("context_loading", "completed", chunks_available=len(retriever.chunks))

    # Stage 2: retrieval
    query_embedding: Optional[list[float]] = None
    candidates: list[tuple[Chunk, float]] = []
    try:
        api_key = os.getenv("JINA_API_KEY", "")
        if api_key and client is not None:
            query_embedding = await _embed_query(question, api_key, client)
        candidates = retriever.search(question, query_embedding, top_k=TOP_K)
        trace.record_stage(
            "retrieval",
            "completed",
            candidates_found=len(candidates),
            top_score=round(candidates[0][1], 4) if candidates else 0,
        )
    except Exception as exc:
        trace.record_error("retrieval", str(exc))

    # Stage 3: reranking
    reranked: list[tuple[Chunk, float]] = []
    top_score = 0.0
    try:
        reranked = await rerank(question, candidates, top_n=TOP_N, client=client)
        top_score = reranked[0][1] if reranked else 0.0
        trace.record_stage(
            "reranking",
            "completed",
            chunks_after_rerank=len(reranked),
            top_score=round(top_score, 4),
        )
    except Exception as exc:
        trace.record_error("reranking", str(exc))
        reranked = candidates[:TOP_N]
        top_score = reranked[0][1] if reranked else 0.0

    is_fallback = top_score < FALLBACK_THRESHOLD or not reranked
    fallback_reason: Optional[str] = (
        "Релевантний контекст не знайдено в базі знань" if is_fallback else None
    )

    trace.context_chunks = [
        {"section": c.section, "text": c.text, "score": round(s, 4)}
        for c, s in reranked
    ]

    # Stage 4: LLM generation
    answer = ""
    try:
        if is_fallback:
            answer = (
                "На основі наданої бази знань неможливо надати точну відповідь на це питання. "
                "У базі знань відсутня достатня інформація для відповіді."
            )
            trace.record_stage("llm_generation", "skipped", reason="fallback_triggered")
        else:
            answer = await generate_answer(question, reranked, client=client)
            trace.record_stage("llm_generation", "completed", answer_length=len(answer))
    except Exception as exc:
        trace.record_error("llm_generation", str(exc))
        answer = "Виникла технічна помилка під час генерації відповіді."
        is_fallback = True
        fallback_reason = f"LLM error: {exc}"

    # Stage 5: validation & response
    trace.record_stage("validation", "completed")
    trace.record_stage("response", "completed")

    confidence = _confidence(top_score, is_fallback)
    trace.answer = answer
    trace.confidence = confidence
    trace.fallback_reason = fallback_reason
    trace.finish()

    # async trace write — does not block response
    asyncio.create_task(asyncio.to_thread(trace.write))

    return AskResponse(
        answer=answer,
        sources=[
            Source(section=c.section, chunk=c.text, score=round(s, 4))
            for c, s in reranked
        ],
        confidence=confidence,
        fallback_reason=fallback_reason,
        trace_id=trace.trace_id,
        latency_ms=trace.latency_ms,
    )
