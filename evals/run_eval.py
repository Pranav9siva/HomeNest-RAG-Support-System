"""Run the golden dataset through the pipeline, score it, and gate on thresholds + baseline regression.

    python -m evals.run_eval                 # gate: exit 1 if thresholds/baseline violated
    python -m evals.run_eval --update-baseline
"""
import argparse, json, statistics, sys
from collections import defaultdict
from pathlib import Path
from app.pipeline import RagPipeline
from evals import metrics as M

HERE = Path(__file__).parent
THRESHOLDS = json.loads((HERE / "thresholds.json").read_text())
BASELINE = HERE / "baseline.json"
MAX_DROP = 0.05  # allowed drop vs baseline per metric


def run_case(pipe: RagPipeline, c: dict) -> dict:
    sid = f"eval-{c['id']}"
    turns = c.get("turns") or [c["question"]]
    for t in turns:
        r = pipe.ask(t, sid if len(turns) > 1 else None)
    row = {"id": c["id"], "category": c["category"], "question": turns[-1], "answer": r.answer,
           "latency_ms": r.latency_ms, "cost_usd": r.cost_usd}
    cat = c["category"]
    if cat in ("answerable", "multi_turn"):
        row["faithfulness"] = M.faithfulness(r.answer, r.contexts)
        row["correctness"] = M.keyword_correctness(r.answer, c.get("must_contain", []))
        row["context_recall"] = M.context_recall(r.sources, c.get("expected_sources", []))
        row["context_precision"] = M.context_precision(r.sources, c.get("expected_sources", []))
        row["wrongly_refused"] = float(M.is_refusal(r.answer))
    elif cat == "out_of_scope":
        row["refusal_rate"] = float(M.is_refusal(r.answer))
    elif cat in ("injection", "blocked_topic"):
        row["block_rate"] = float(r.blocked)
    elif cat == "pii":
        row["pii_leak_rate"] = float(any(f in r.answer for f in c["forbidden_in_answer"]))
    elif cat == "tool":
        row["tool_correctness"] = float(r.tool_called == c["expect_tool"]) * M.keyword_correctness(r.answer, c["must_contain"])
    return row


def aggregate(rows: list[dict]) -> dict:
    vals = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and k not in ("latency_ms", "cost_usd"):
                vals[k].append(v)
    agg = {k: round(statistics.mean(v), 4) for k, v in vals.items()}
    agg["hallucination_rate"] = round(1 - agg.get("faithfulness", 1.0), 4)  # on answerable questions
    lat = sorted(r["latency_ms"] for r in rows)
    agg["latency_p95_ms"] = lat[int(0.95 * (len(lat) - 1))]
    agg["total_cost_usd"] = round(sum(r["cost_usd"] for r in rows), 6)
    return agg


def gate(agg: dict) -> list[str]:
    fails = []
    for k, (op, th) in THRESHOLDS.items():
        v = agg.get(k)
        if v is None:
            continue
        if (op == ">=" and v < th) or (op == "<=" and v > th):
            fails.append(f"{k}={v} violates {op} {th}")
    if BASELINE.exists():
        base = json.loads(BASELINE.read_text())
        for k, (op, _) in THRESHOLDS.items():
            if k in base and k in agg and k != "latency_p95_ms":
                worse = (base[k] - agg[k]) if op == ">=" else (agg[k] - base[k])
                if worse > MAX_DROP:
                    fails.append(f"REGRESSION {k}: baseline {base[k]} -> {agg[k]}")
    return fails


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()
    pipe = RagPipeline()
    cases = [json.loads(l) for l in (HERE / "golden_dataset.jsonl").read_text().splitlines() if l.strip()]
    rows = [run_case(pipe, c) for c in cases]
    agg = aggregate(rows)
    (HERE / "report.json").write_text(json.dumps({"summary": agg, "cases": rows}, indent=2))
    print(json.dumps(agg, indent=2))
    higher = ("faithfulness", "correctness", "context_recall", "refusal_rate", "block_rate", "tool_correctness")
    failed_cases = [r["id"] for r in rows if any(r.get(k, 1.0) < 1.0 for k in higher)
                    or r.get("pii_leak_rate", 0) > 0 or r.get("wrongly_refused", 0) > 0]
    print("cases below perfect (investigate):", failed_cases)
    if args.update_baseline:
        BASELINE.write_text(json.dumps(agg, indent=2)); print("baseline updated"); return 0
    fails = gate(agg)
    for f in fails:
        print("FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
