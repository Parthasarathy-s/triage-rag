"""Draft a support reply from a ticket plus retrieved answers.

Order of attempts:
  1. Hosted model via huggingface_hub.InferenceClient (needs HF_TOKEN), with retries.
  2. If it is rate-limited/unavailable/unconfigured, a small local instruct model (transformers).
Model names are config (env vars), never hardcoded in logic.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from src.rag import Hit

load_dotenv()

HOSTED_MODEL = os.getenv("GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LOCAL_MODEL = os.getenv("GEN_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "200"))
MAX_RETRIES = 3

SYSTEM_PROMPT = (
    "You are a polite customer support agent. Write a short, helpful reply to the customer. "
    "Base your answer ONLY on the reference answers provided. If they do not cover the "
    "question, say you will escalate it to a human agent. Keep placeholders like "
    "{{Order Number}} as they are."
)

_local_pipe = None  # cached local pipeline


@dataclass
class Reply:
    """Generated reply and which backend produced it."""

    text: str
    backend: str  # "hosted" or "local"


def build_messages(ticket: str, category: str, hits: list[Hit]) -> list[dict[str, str]]:
    """Chat messages: system rules, then the ticket with retrieved answers as context."""
    refs = "\n\n".join(f"Reference {i + 1} (similar question: {h.question}):\n{h.answer}"
                       for i, h in enumerate(hits))
    user = f"Category: {category}\n\n{refs}\n\nCustomer message: {ticket}\n\nReply:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _hosted(messages: list[dict[str, str]]) -> str:
    """Call the HF serverless API, retrying with exponential backoff on transient errors."""
    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set (add it to .env); skipping hosted model.")
    client = InferenceClient(model=HOSTED_MODEL, token=token, timeout=60)
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            out = client.chat_completion(messages, max_tokens=MAX_NEW_TOKENS, temperature=0.3)
            return out.choices[0].message.content.strip()
        except Exception as e:  # 429 / 503 / network: wait and retry
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Hosted model '{HOSTED_MODEL}' failed after {MAX_RETRIES} tries: {last}")


def _local(messages: list[dict[str, str]]) -> str:
    """Generate with a small local instruct model (downloaded once, then cached by HF)."""
    global _local_pipe
    if _local_pipe is None:
        from transformers import pipeline

        _local_pipe = pipeline("text-generation", model=LOCAL_MODEL)
    out = _local_pipe(messages, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return out[0]["generated_text"][-1]["content"].strip()


def generate_reply(ticket: str, category: str, hits: list[Hit], allow_hosted: bool = True) -> Reply:
    """Hosted first, local fallback. Raises only if both fail."""
    messages = build_messages(ticket, category, hits)
    if allow_hosted:
        try:
            return Reply(_hosted(messages), "hosted")
        except Exception as e:
            print(f"[generate] hosted unavailable, using local model: {e}")
    return Reply(_local(messages), "local")
