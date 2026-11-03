"""Multi-turn workflows, tool calling and failure recovery."""
import pytest

def test_followup_uses_previous_turn(pipe):
    pipe.ask("How many speed levels does the Aero have?", "mt-a")
    r = pipe.ask("and what sizes does it come in?", "mt-a")
    assert "1400" in r.answer

def test_sessions_are_isolated(pipe):
    pipe.ask("How many speed levels does the Aero have?", "iso-1")
    pipe.ask("Is phone support available?", "iso-2")
    assert len(pipe._rewrite("and what sizes?", pipe.sessions["iso-2"]).split()) > 3
    assert "speed levels" not in pipe._rewrite("and what sizes?", pipe.sessions["iso-2"])
    assert "speed levels" in pipe._rewrite("and what sizes?", pipe.sessions["iso-1"])

def test_history_recorded(pipe):
    pipe.ask("Is phone support available?", "hist")
    assert [m["role"] for m in pipe.sessions["hist"]] == ["user", "assistant"]

@pytest.mark.parametrize("q,expect", [
    ("Where is my order HN1002?", "Out for delivery"),
    ("track order hn1001", "Delivered"),
    ("Where is my order HN1003", "Processing"),
])
def test_tool_call_success(pipe, q, expect):
    r = pipe.ask(q)
    assert r.tool_called == "order_status" and expect in r.answer

def test_tool_failure_recovery_unknown_order(pipe):
    r = pipe.ask("Where is my order HN9999?")
    assert r.tool_called == "order_status" and "not found" in r.answer and "support" in r.answer.lower()

def test_tool_missing_argument_asks_user(pipe):
    r = pipe.ask("Where is my order?")
    assert "order id" in r.answer.lower()

def test_production_trace_logged(pipe, tmp_path):
    """Every request is logged with latency/cost so failures can be promoted to regression tests."""
    import json
    from app import config
    pipe.ask("Is phone support available?")
    last = json.loads(config.LOG_PATH.read_text().splitlines()[-1])
    assert {"question", "answer", "latency_ms", "cost_usd", "retrieval_score"} <= set(last)
