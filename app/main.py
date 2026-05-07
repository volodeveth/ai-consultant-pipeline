from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

load_dotenv()

from app.models import AskRequest, AskResponse
from app.pipeline import close_pipeline, init_pipeline, process_question

KB_PATH = Path("data/knowledge_base.md")


@asynccontextmanager
async def lifespan(app: FastAPI):
    count = await init_pipeline(KB_PATH)
    print(f"[startup] Indexed {count} chunks from {KB_PATH}")
    yield
    await close_pipeline()


app = FastAPI(title="AI Consultant API", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    return await process_question(request.question)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
