import json
import re
import time
from dataclasses import dataclass, field
from . import config, guardrails, tools
from .llm import REFUSAL, coverage, get_llm
from .retriever import Retriever

FOLLOWUP_HINTS = ("it", "that", "this", "they", "those", "them", "its", "and", "also")


@dataclass
class Result:
    answer: str
    sources: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None
    tool_called: str | None = None
    pii_redacted: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    retrieval_score: float = 0.0


class RagPipeline:
    def __init__(self, llm=None):
        self.retriever = Retriever()
        self.n_chunks = self.retriever.ingest_dir()
        self.llm = llm or get_llm()
        self.sessions: dict[str, list[dict]] = {}

    def _rewrite(self, q: str, history: list[dict]) -> str:
        """Resolve follow-ups ('what about it?') by prepending the previous user question."""
        words = q.lower().split()
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev and (len(words) <= 4 or any(w in FOLLOWUP_HINTS for w in words[:2])):
            concise = re.sub(r"^\s*(?:what\s+about|and\s+what\s+about)\s+", "", q, flags=re.I).strip()
            return f"{prev[-1]} {concise or q}"
        return q

    def ask(self, question: str, session_id: str | None = None) -> Result:
        t0 = time.perf_counter()
        history = self.sessions.setdefault(session_id, []) if session_id else []
        ok, reason = guardrails.check_input(question)
        if not ok:
            res = Result(answer="Sorry, I can't help with that request.", blocked=True, block_reason=reason)
            return self._finish(res, t0, question, session_id, history, {})
        clean_q, pii = guardrails.redact_pii(question)

        tool = tools.maybe_call_tool(clean_q)
        if tool:
            name, data, err = tool
            ans = err if err else f"Order {data['order_id']} status: {data['status']}" + (
                f" (ETA {data['eta']})" if "eta" in data else f" (delivered on {data['delivered_on']})")
            res = Result(answer=ans, tool_called=name, pii_redacted=pii)
            return self._finish(res, t0, clean_q, session_id, history, {})

        rq = self._rewrite(clean_q, history)
        chunks = self.retriever.search(rq)
        top = chunks[0].score if chunks else 0.0
        if top < config.MIN_RETRIEVAL_SCORE or coverage(clean_q, chunks) < config.MIN_COVERAGE:
            res = Result(answer=REFUSAL, retrieval_score=top, pii_redacted=pii)
            return self._finish(res, t0, clean_q, session_id, history, {})
        try:
            answer, usage = self.llm.generate(clean_q, chunks, history)  # history resolves follow-ups in extractive answering
        except Exception:  # failure recovery: degrade gracefully instead of 500
            answer, usage = "I'm having trouble right now. Please try again or contact support.", {}
        answer, out_pii = guardrails.redact_pii(answer)  # output-side PII guard
        res = Result(answer=answer, sources=sorted({c.source for c in chunks}),
                     contexts=[c.text for c in chunks], retrieval_score=top, pii_redacted=pii + out_pii)
        return self._finish(res, t0, clean_q, session_id, history, usage)

    def _finish(self, res: Result, t0, q, session_id, history, usage):
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        res.cost_usd = round(usage.get("prompt_tokens", 0) / 1000 * config.PRICE_IN
                             + usage.get("completion_tokens", 0) / 1000 * config.PRICE_OUT, 6)
        if session_id is not None:
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": res.answer}]
        config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.LOG_PATH, "a") as f:  # production traces -> turn failures into regression tests
            f.write(json.dumps({"ts": time.time(), "question": q, "answer": res.answer, "blocked": res.block_reason,
                                "tool": res.tool_called, "latency_ms": res.latency_ms, "cost_usd": res.cost_usd,
                                "retrieval_score": res.retrieval_score, "usage": usage}) + "\n")
        return res
