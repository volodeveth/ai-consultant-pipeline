import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

TRACES_FILE = Path("traces.jsonl")


@dataclass
class PipelineTrace:
    trace_id: str
    question: str
    stages: dict = field(default_factory=dict)
    context_chunks: list = field(default_factory=list)
    answer: str = ""
    confidence: str = ""
    fallback_reason: Optional[str] = None
    latency_ms: int = 0
    errors: list = field(default_factory=list)
    _started_at: float = field(default_factory=time.time, repr=False)

    def record_stage(self, name: str, status: str, **kwargs: object) -> None:
        self.stages[name] = {"status": status, **kwargs}

    def record_error(self, stage: str, error: str) -> None:
        self.errors.append({"stage": stage, "error": error})
        self.stages[stage] = {"status": "error", "error": error}

    def finish(self) -> None:
        self.latency_ms = int((time.time() - self._started_at) * 1000)

    def write(self) -> None:
        entry = {
            "trace_id": self.trace_id,
            "question": self.question,
            "pipeline_stages": self.stages,
            "context_chunks": self.context_chunks,
            "answer": self.answer,
            "confidence": self.confidence,
            "fallback_reason": self.fallback_reason,
            "latency_ms": self.latency_ms,
            "errors": self.errors,
        }
        with open(TRACES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def new_trace(question: str) -> PipelineTrace:
    return PipelineTrace(trace_id=str(uuid.uuid4()), question=question)
