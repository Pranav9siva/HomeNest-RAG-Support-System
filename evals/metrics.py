"""Offline, deterministic RAG metrics (no LLM judge needed) so CI is fast, free and reproducible.
They are proxies; evals/ragas_eval.py and evals/deepeval_eval.py add LLM-judged versions."""
import re
from app.llm import REFUSAL
from app.retriever import analyzer


def _content(s: str) -> set[str]:
    return {t for t in analyzer(s) if "_" not in t}


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Share of answer sentences whose content words are >=80% present in the retrieved context (groundedness)."""
    ctx = set().union(*[_content(c) for c in contexts]) if contexts else set()
    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer) if len(s) > 3]
    if not sents:
        return 1.0
    ok = 0
    for s in sents:
        w = _content(s)
        ok += (len(w & ctx) / len(w) >= 0.8) if w else 1
    return ok / len(sents)


def keyword_correctness(answer: str, must_contain: list[str]) -> float:
    a = answer.lower()
    return sum(k.lower() in a for k in must_contain) / len(must_contain) if must_contain else 1.0


def context_recall(sources: list[str], expected: list[str]) -> float:
    return sum(e in sources for e in expected) / len(expected) if expected else 1.0


def context_precision(sources: list[str], expected: list[str]) -> float:
    return sum(s in expected for s in sources) / len(sources) if sources else 0.0


def is_refusal(answer: str) -> bool:
    return answer.strip() == REFUSAL
