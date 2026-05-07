from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.models import AskRequest, AskResponse
from app.pipeline import init_pipeline, process_question

KB_PATH = Path("data/knowledge_base.md")


@asynccontextmanager
async def lifespan(app: FastAPI):
    count = await init_pipeline(KB_PATH)
    print(f"[startup] Indexed {count} chunks from {KB_PATH}")
    yield


app = FastAPI(title="AI Consultant API", version="1.0.0", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    return await process_question(request.question)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
