"""REST API contract tests (schema / OpenAPI / validation)."""

def test_health(client):
    j = client.get("/health").json()
    assert j["status"] == "ok" and j["chunks"] > 0

def test_openapi_documents_chat(client):
    spec = client.get("/openapi.json").json()
    assert "/chat" in spec["paths"] and "post" in spec["paths"]["/chat"]
    props = spec["components"]["schemas"]["ChatResponse"]["properties"]
    for f in ("answer", "sources", "blocked", "latency_ms", "cost_usd"):
        assert f in props

def test_chat_response_shape(client):
    r = client.post("/chat", json={"question": "How long is the warranty on the fan motor?"})
    assert r.status_code == 200
    j = r.json()
    assert "2-year" in j["answer"] and "warranty" in j["sources"] and j["latency_ms"] >= 0

def test_validation_errors(client):
    assert client.post("/chat", json={}).status_code == 422
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "x" * 1001}).status_code == 422

def test_metrics_endpoint_counts_requests(client):
    before = client.get("/metrics").json()["requests"]
    client.post("/chat", json={"question": "Is phone support available?"})
    after = client.get("/metrics").json()
    assert after["requests"] == before + 1 and after["avg_latency_ms"] >= 0
