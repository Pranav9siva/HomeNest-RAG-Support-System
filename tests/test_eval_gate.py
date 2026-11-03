"""Golden-dataset gate. This is the pre-release check that CI runs."""
import json
from evals import run_eval
from app.pipeline import RagPipeline

def _run(pipe):
    cases = [json.loads(l) for l in (run_eval.HERE / "golden_dataset.jsonl").read_text().splitlines() if l.strip()]
    return run_eval.aggregate([run_eval.run_case(pipe, c) for c in cases])

def test_golden_dataset_meets_thresholds(pipe):
    assert run_eval.gate(_run(pipe)) == []

def test_gate_catches_a_degraded_pipeline():
    """Sanity check on the eval itself: a model that always hallucinates must FAIL the gate."""
    class Liar:
        name = "liar"
        def generate(self, q, ctx, hist):
            return "Our products are guaranteed for 50 years and we offer phone support 24/7.", {}
    fails = run_eval.gate(_run(RagPipeline(llm=Liar())))
    assert any("faithfulness" in f or "hallucination" in f for f in fails)
    assert any("correctness" in f for f in fails)

def test_golden_dataset_is_well_formed():
    ids = []
    for l in (run_eval.HERE / "golden_dataset.jsonl").read_text().splitlines():
        c = json.loads(l); ids.append(c["id"])
        assert c["category"] and ("question" in c or "turns" in c)
    assert len(ids) == len(set(ids)) and len(ids) >= 30
