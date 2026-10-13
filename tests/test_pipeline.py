from fastapi.testclient import TestClient

import src.api as api
from src.generate import Reply, build_messages
from src.rag import Hit

HIT = Hit(question="where is my order", answer="Check the tracking page.",
          label="shipping", intent="track_order", distance=0.1)


def test_build_messages_contains_ticket_and_reference():
    msgs = build_messages("Where is my parcel?", "shipping", [HIT])
    assert msgs[0]["role"] == "system"
    assert "Where is my parcel?" in msgs[1]["content"]
    assert "Check the tracking page." in msgs[1]["content"]


class FakeClassifier:
    def predict(self, texts):
        return [("shipping", 0.97) for _ in texts]


def test_triage_endpoint_with_stubbed_models(monkeypatch):
    monkeypatch.setattr(api, "get_classifier", lambda: FakeClassifier())
    monkeypatch.setattr(api, "retrieve", lambda q, k=3, label=None: [HIT])
    monkeypatch.setattr(api, "generate_reply", lambda *a, **k: Reply("On its way.", "local"))
    client = TestClient(api.app)
    r = client.post("/triage", json={"ticket": "Where is my parcel?"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "shipping" and body["reply"] == "On its way."
    assert body["sources"][0]["answer"] == "Check the tracking page."


def test_triage_rejects_too_short_ticket():
    assert TestClient(api.app).post("/triage", json={"ticket": "a"}).status_code == 422


def test_health():
    assert TestClient(api.app).get("/health").json() == {"status": "ok"}
