# Support Ticket Triage and Answer Assistant

Portfolio project for AI Engineer roles, and I am also using it to learn. Explain key steps briefly in comments and in your summaries.

## Workflow rules
- Work one phase at a time. After each phase: summarize what changed, show the exact command to run it, then stop and wait for my go-ahead.
- Ask before major decisions (dataset choice, model choice).
- Never invent metrics. Only report numbers from runs that were actually executed.
- Never hardcode secrets. Use .env (HF_TOKEN) and keep .env.example up to date.
- Pin versions in requirements.txt. Use typed, small functions with docstrings.
- Everything must work on free tiers. No paid APIs.

## Stack
Python 3.11, pandas, scikit-learn, Hugging Face Transformers/Datasets, huggingface_hub (InferenceClient), sentence-transformers, ChromaDB, FastAPI, Streamlit, Docker, pytest.

## Free Hugging Face usage
- LLM reply generation: huggingface_hub InferenceClient with HF_TOKEN, using a small instruct model that is currently available on the free serverless tier. Check current availability before choosing, and make the model name a config value, not hardcoded.
- Fallback if the hosted model is rate-limited or unavailable: a small local instruct model (around 1B parameters) through transformers.
- Zero-shot classification baseline: use the local facebook/bart-large-mnli pipeline (free, no API needed), or the hosted LLM if it works well.
- Embeddings: sentence-transformers/all-MiniLM-L6-v2, run locally.
- Handle rate limits with retries and clear error messages.

## Structure
data/, notebooks/, src/ (data.py, baseline.py, finetune.py, evaluate.py, rag.py, generate.py, api.py), app.py (Streamlit), tests/, Dockerfile, requirements.txt, README.md

## Deployment target
Hugging Face Spaces (Streamlit). The fine-tuned model goes on the HF Hub.