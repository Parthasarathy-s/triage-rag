"""Retrieval over past support answers (RAG knowledge base) with ChromaDB.

The knowledge base is the (question -> answer) pairs from the *training split only*, so
evaluation tickets never leak into retrieval. We embed the past QUESTION, because a new
ticket is most similar to questions people asked before, and return the stored ANSWER.

Run:  python -m src.rag build        build the index (once)
      python -m src.rag "where is my order?"   try a query
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb

from src.data import DATASET_ID, INTENT_TO_LABEL, load_raw, load_split, normalize_text

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma_db"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION = "support_answers"

_embedder = None  # loaded lazily and reused (loading the model is the slow part)


@dataclass
class Hit:
    """One retrieved past ticket."""

    question: str
    answer: str
    label: str
    intent: str
    distance: float  # cosine distance: lower = more similar


def embed(texts: list[str]):
    """Embed texts with MiniLM; normalized vectors so cosine distance is meaningful."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(texts, batch_size=128, normalize_embeddings=True).tolist()


def _collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # first build: nothing to delete
            pass
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build_index() -> int:
    """(Re)build the vector index from train-split questions and their dataset answers."""
    raw = load_raw()
    raw["text"] = raw["instruction"].map(normalize_text)
    train_texts = set(load_split("train")["text"])
    kb = raw[raw["text"].isin(train_texts)].drop_duplicates("text").reset_index(drop=True)
    kb["label"] = kb["intent"].map(INTENT_TO_LABEL)

    col = _collection(reset=True)
    step = 2000  # add in chunks to keep memory and request sizes small
    for i in range(0, len(kb), step):
        part = kb.iloc[i : i + step]
        col.add(
            ids=[str(j) for j in part.index],
            embeddings=embed(part["text"].tolist()),
            documents=part["text"].tolist(),
            metadatas=[{"answer": a, "label": l, "intent": t}
                       for a, l, t in zip(part["response"], part["label"], part["intent"])],
        )
    return col.count()


def retrieve(query: str, k: int = 3, label: str | None = None) -> list[Hit]:
    """Top-k similar past tickets. `label` restricts search to the predicted category."""
    col = _collection()
    if col.count() == 0:
        raise RuntimeError("Vector index is empty. Run:  python -m src.rag build")
    res = col.query(
        query_embeddings=embed([query]),
        n_results=k,
        where={"label": label} if label else None,
    )
    return [
        Hit(question=d, answer=m["answer"], label=m["label"], intent=m["intent"], distance=dist)
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print(f"indexed {build_index()} tickets from {DATASET_ID} (train split)")
    else:
        for h in retrieve(" ".join(sys.argv[1:]) or "where is my order?"):
            print(f"[{h.distance:.3f}] ({h.label}/{h.intent}) {h.question}\n    -> {h.answer[:160]}")
