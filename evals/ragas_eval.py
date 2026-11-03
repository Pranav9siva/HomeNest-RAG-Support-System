"""LLM-judged RAG metrics with Ragas (faithfulness, answer relevancy, context precision/recall).

    pip install -r requirements-eval.txt
    export LLM_API_KEY=...            # judge model credentials (OpenAI-compatible)
    python -m evals.ragas_eval
Exits non-zero if any mean score is below its threshold, so it can gate a release.
"""
import os, sys
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
from evals._judge_common import answerable_rows, require_key

THRESHOLDS = {"faithfulness": 0.85, "answer_relevancy": 0.75, "context_recall": 0.80}


def main() -> int:
    require_key()
    base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm = LangchainLLMWrapper(ChatOpenAI(model=os.getenv("LLM_JUDGE_MODEL", "gpt-4o-mini"), base_url=base,
                                         api_key=os.environ["LLM_API_KEY"], temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(base_url=base, api_key=os.environ["LLM_API_KEY"]))
    ds = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"], "retrieved_contexts": r["contexts"], "reference": r["ground_truth"]}
        for r in answerable_rows()])
    res = evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
                   llm=llm, embeddings=emb)
    df = res.to_pandas()
    df.to_csv("evals/ragas_report.csv", index=False)
    means = df.select_dtypes("number").mean().to_dict()
    print(means)
    bad = {k: means[k] for k, t in THRESHOLDS.items() if k in means and means[k] < t}
    for k, v in bad.items():
        print(f"FAIL {k}={v:.3f} < {THRESHOLDS[k]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
