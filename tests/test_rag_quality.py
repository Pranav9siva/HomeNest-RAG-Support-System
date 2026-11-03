"""Hallucination / refusal / retrieval behaviour on individual cases."""
from app.llm import REFUSAL

def test_out_of_scope_refused(pipe):
    for q in ["What is the capital of France?", "Tell me a joke about cats", "What is the price of the Aero fan in dollars?"]:
        assert pipe.ask(q).answer == REFUSAL

def test_answers_are_grounded_in_context(pipe):
    from evals.metrics import faithfulness
    r = pipe.ask("What is the recommended operating voltage range?")
    assert faithfulness(r.answer, r.contexts) == 1.0 and "285" in r.answer

def test_no_invented_numbers(pipe):
    """Every number in the answer must appear in the retrieved context."""
    import re
    r = pipe.ask("How many watts does the Aero consume at top speed?")
    ctx = " ".join(r.contexts)
    for n in re.findall(r"\d+", r.answer):
        assert n in ctx

def test_retrieval_recall_on_each_doc(pipe):
    cases = {"warranty": "How do I claim warranty?", "returns": "When will I get my refund?",
             "wifi_setup": "Why can't I see my wifi network?", "troubleshooting": "The remote does not respond",
             "energy": "What is the speed of the sleep mode?", "support": "Do you offer phone support?"}
    for src, q in cases.items():
        assert src in pipe.ask(q).sources, (src, q)

def test_llm_failure_recovery(pipe):
    class Boom:
        name = "boom"
        def generate(self, *a, **k):
            raise RuntimeError("provider down")
    from app.pipeline import RagPipeline
    p = RagPipeline(llm=Boom())
    r = p.ask("How long is the warranty on the motor?")
    assert "try again" in r.answer.lower() and not r.blocked
