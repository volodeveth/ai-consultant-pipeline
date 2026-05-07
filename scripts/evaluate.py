#!/usr/bin/env python3
"""Auto-evaluation script — runs all test_questions.json against POST /ask."""

import asyncio
import json
from pathlib import Path

import httpx

QUESTIONS_PATH = Path("data/test_questions.json")
BASE_URL = "http://localhost:8000"


async def run() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    results = []

    async with httpx.AsyncClient(timeout=120) as client:
        for q in questions:
            print(f"\n[{q['id']}] {q['question']}")
            resp = await client.post(
                f"{BASE_URL}/ask",
                json={"question": q["question"]},
            )
            data = resp.json()
            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "answer": data["answer"],
                    "confidence": data["confidence"],
                    "fallback_reason": data["fallback_reason"],
                    "top_source": data["sources"][0] if data["sources"] else None,
                    "latency_ms": data["latency_ms"],
                }
            )
            print(f"  confidence : {data['confidence']}")
            print(f"  fallback   : {data['fallback_reason']}")
            print(f"  latency    : {data['latency_ms']} ms")
            print(f"  answer     : {data['answer'][:120]}...")

    out = Path("evaluation_results.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    fallbacks = sum(1 for r in results if r["fallback_reason"])
    print(f"\n{'='*60}")
    print(f"Total: {len(results)} | Answered: {len(results)-fallbacks} | Fallbacks: {fallbacks}")
    print(f"Results saved to {out}")


if __name__ == "__main__":
    asyncio.run(run())
