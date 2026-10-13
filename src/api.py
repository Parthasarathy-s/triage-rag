"""FastAPI service + the shared `triage` pipeline (also used by the Streamlit app).

Run:  uvicorn src.api:app --reload      then open http://127.0.0.1:8000/docs
Pipeline: classify -> retrieve similar past answers (filtered to that label) -> draft reply.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.finetune import TicketClassifier
from src.generate import generate_reply
from src.rag import retrieve


class TicketIn(BaseModel):
    """Request body."""

    ticket: str = Field(min_length=3, max_length=2000)
    generate: bool = True  # set false to only classify + retrieve (fast, no LLM)


class SourceOut(BaseModel):
    question: str
    answer: str
    distance: float


class TriageOut(BaseModel):
    """Response body."""

    label: str
    confidence: float
    sources: list[SourceOut]
    reply: str | None
    reply_backend: str | None


@lru_cache(maxsize=1)
def get_classifier() -> TicketClassifier:
    """Load the classifier once per process."""
    return TicketClassifier()


def triage(ticket: str, generate: bool = True, k: int = 3) -> TriageOut:
    """Run the full pipeline for one ticket."""
    label, conf = get_classifier().predict([ticket])[0]
    hits = retrieve(ticket, k=k, label=label)
    reply = backend = None
    if generate:
        r = generate_reply(ticket, label, hits)
        reply, backend = r.text, r.backend
    return TriageOut(
        label=label, confidence=conf, reply=reply, reply_backend=backend,
        sources=[SourceOut(question=h.question, answer=h.answer, distance=h.distance) for h in hits],
    )


app = FastAPI(title="Support Ticket Triage", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/triage", response_model=TriageOut)
def triage_endpoint(body: TicketIn) -> TriageOut:
    """Classify a ticket and draft a grounded reply."""
    try:
        return triage(body.ticket, body.generate)
    except RuntimeError as e:  # e.g. empty vector index or no model available
        raise HTTPException(status_code=503, detail=str(e)) from e
