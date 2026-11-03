# RAG Support Assistant with Automated LLM Evaluation

A retrieval-augmented support assistant (fictional "HomeNest" smart-fan knowledge base) built to be **tested**:
a FastAPI service plus an evaluation harness that scores groundedness, hallucination, refusal behaviour,
guardrails and tool calling, and blocks regressions in CI.

```
question ─► input guardrails ─► tool router ─► retriever (TF-IDF vectors in ChromaDB, top-k)
                │ blocked                          │ low score / low coverage ─► refusal
                ▼                                  ▼
          safe refusal                      LLM (offline extractive | any OpenAI-compatible API)
                                                   ▼
                                       output PII redaction ─► JSON trace log (latency, cost)
```

## Quick start
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload          # Swagger UI at http://localhost:8000/docs
curl -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"question":"Does the device work on 5GHz wifi?","session_id":"demo"}'
```
Default LLM is a deterministic **extractive** model (no key, reproducible CI). To use a real LLM:
```bash
export LLM_PROVIDER=openai LLM_API_KEY=... LLM_MODEL=gpt-4o-mini   # LLM_BASE_URL for Groq etc.
```

## What is tested
| Layer | Where | What |
|---|---|---|
| API contract | `tests/test_api.py` | OpenAPI schema, request validation (422s), response shape, `/metrics` |
| Guardrails | `tests/test_guardrails.py` | prompt injection, jailbreaks, blocked topics, PII redaction (phone, email, card, Aadhaar) |
| RAG quality | `tests/test_rag_quality.py` | grounding, no invented numbers, out-of-scope refusal, per-doc retrieval recall, LLM-outage recovery |
| Conversation / tools | `tests/test_conversation_and_tools.py` | multi-turn follow-ups, session isolation, tool call success / unknown-id / missing-arg recovery, trace logging |
| Golden dataset gate | `evals/run_eval.py`, `tests/test_eval_gate.py` | 35 cases; thresholds + baseline-regression check; proves the gate **fails** on a hallucinating model |
| Red-team (black box) | `promptfoo/promptfooconfig.yaml` | 15 deterministic HTTP-level assertions against the live API |
| LLM-judged | `evals/ragas_eval.py`, `evals/deepeval_eval.py` | Ragas faithfulness / relevancy / context precision+recall; DeepEval faithfulness / relevancy / recall |

```bash
python -m pytest -q                       # 36 tests
python -m evals.run_eval                  # offline gate, exit 1 on violation
python -m evals.run_eval --update-baseline
npx promptfoo@latest eval -c promptfoo/promptfooconfig.yaml      # API must be running
pip install -r requirements-eval.txt && python -m evals.ragas_eval  # needs LLM_API_KEY
```

## Metrics (offline gate)
`faithfulness` (share of answer sentences supported by retrieved context; `hallucination_rate = 1 - faithfulness`),
`correctness` (required facts present, not exact string match), `context_recall/precision`, `refusal_rate`
(out-of-scope), `block_rate` (injection/blocked topics), `pii_leak_rate`, `tool_correctness`, `latency_p95_ms`, cost.
Because LLM output is non-deterministic, scoring is metric-threshold based, not pass/fail on exact text.
Thresholds live in `evals/thresholds.json`; a >0.05 drop vs `evals/baseline.json` also fails the build.

## Monitoring and production-failure loop
Every request appends a JSON line to `logs/requests.jsonl` (question, answer, latency, token cost, retrieval score,
block reason, tool). To turn a production failure into a regression test, copy the line into
`evals/golden_dataset.jsonl` with its expected answer/sources and re-run the gate.

## Known limitations (honest baseline)
- Two golden cases are still below perfect: `ans-17` ("support **hours**" vs "9 AM to 6 PM": a vocabulary gap TF-IDF cannot bridge)
  and `mt-02` (rule-based follow-up rewriting picks the wrong sentence). A real embedding model and an LLM-based
  query rewriter are the intended fixes; the gate will show whether they help.
- The golden set was used while tuning the offline pipeline, so offline scores are optimistic. Add held-out cases before trusting them.
- Guardrails are regex based (fast, auditable, bypassable). Production systems should layer a classifier/LLM guard; promptfoo's
  `redteam` generators are a good way to find bypasses.
- The corpus is small and fictional, and the offline LLM is extractive, so hallucination is ~0 by construction there.
  Run the same suite with `LLM_PROVIDER=openai` to measure a generative model.
- Ragas/DeepEval scripts need a judge LLM key and have not been run against a judge; pinned to ragas 0.2.15
  (0.4.x failed to import against current langchain-community).
