"""DeepEval metrics (G-Eval style judges) as a pytest suite.

    pip install -r requirements-eval.txt
    export LLM_API_KEY=...   # DeepEval uses OPENAI_API_KEY; we map LLM_API_KEY to it below
    deepeval test run evals/deepeval_eval.py        # or: pytest evals/deepeval_eval.py
"""
import os
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from evals._judge_common import answerable_rows

if not os.getenv("LLM_API_KEY"):
    pytest.skip("LLM_API_KEY not set; LLM-judged evals skipped", allow_module_level=True)
os.environ.setdefault("OPENAI_API_KEY", os.environ["LLM_API_KEY"])

ROWS = answerable_rows()


@pytest.mark.parametrize("row", ROWS, ids=[r["question"][:40] for r in ROWS])
def test_rag_answer(row):
    case = LLMTestCase(input=row["question"], actual_output=row["answer"], expected_output=row["ground_truth"],
                       retrieval_context=row["contexts"])
    assert_test(case, [FaithfulnessMetric(threshold=0.8), AnswerRelevancyMetric(threshold=0.7),
                       ContextualRecallMetric(threshold=0.7)])
