"""Shared helpers for the LLM-judged evals (need LLM_API_KEY; any OpenAI-compatible endpoint)."""
import json, os, sys
from pathlib import Path
from app.pipeline import RagPipeline

DATASET = Path(__file__).parent / "golden_dataset.jsonl"


def require_key():
    if not os.getenv("LLM_API_KEY"):
        sys.exit("Set LLM_API_KEY (and optionally LLM_BASE_URL / LLM_JUDGE_MODEL) to run LLM-judged evals. "
                 "The offline gate (python -m evals.run_eval) needs no key.")


def answerable_rows(limit: int | None = None) -> list[dict]:
    """Run answerable golden cases through the pipeline; return rows with question/answer/contexts/ground truth."""
    pipe = RagPipeline()
    rows = []
    for line in DATASET.read_text().splitlines():
        c = json.loads(line)
        if c["category"] != "answerable":
            continue
        r = pipe.ask(c["question"])
        rows.append({"question": c["question"], "answer": r.answer, "contexts": r.contexts, "ground_truth": c["ground_truth"]})
    return rows[:limit] if limit else rows
