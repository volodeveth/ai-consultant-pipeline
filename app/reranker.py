import os

import httpx

from app.indexer import Chunk

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


async def rerank(
    query: str,
    candidates: list[tuple[Chunk, float]],
    top_n: int = 3,
) -> list[tuple[Chunk, float]]:
    if not candidates:
        return []

    api_key = os.getenv("JINA_API_KEY", "")
    if not api_key or len(candidates) <= top_n:
        return candidates[:top_n]

    documents = [c.text for c, _ in candidates]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "model": "jina-reranker-v3",
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return [
        (candidates[r["index"]][0], float(r["relevance_score"]))
        for r in data["results"]
    ]
