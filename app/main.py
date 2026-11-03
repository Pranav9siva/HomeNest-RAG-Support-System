from contextlib import asynccontextmanager
from dataclasses import asdict
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .pipeline import RagPipeline

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipe"] = RagPipeline()
    state["stats"] = {"requests": 0, "blocked": 0, "latency_ms_total": 0.0, "cost_usd_total": 0.0}
    yield


app = FastAPI(title="HomeNest RAG Support Assistant", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(None, description="Pass the same id to keep multi-turn context")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    contexts: list[str]
    blocked: bool
    block_reason: str | None
    tool_called: str | None
    pii_redacted: list[str]
    latency_ms: float
    cost_usd: float
    retrieval_score: float


@app.get("/health")
def health():
    return {"status": "ok", "chunks": state["pipe"].n_chunks, "llm": state["pipe"].llm.name}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    r = state["pipe"].ask(req.question, req.session_id)
    s = state["stats"]
    s["requests"] += 1; s["blocked"] += int(r.blocked)
    s["latency_ms_total"] += r.latency_ms; s["cost_usd_total"] += r.cost_usd
    return asdict(r)


@app.get("/metrics")
def metrics():
    s = state["stats"]; n = max(s["requests"], 1)
    return {"requests": s["requests"], "blocked": s["blocked"],
            "avg_latency_ms": round(s["latency_ms_total"] / n, 2), "total_cost_usd": round(s["cost_usd_total"], 6)}
