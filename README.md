# Support Ticket Triage and Answer Assistant

Classifies a customer support ticket (account / billing / general / shipping), retrieves similar past answers from a vector index, and drafts a grounded reply with a small LLM.

```
ticket ─▶ fine-tuned DistilBERT ─▶ label ─┐
   │                                      ▼
   └──▶ MiniLM embedding ─▶ ChromaDB (filtered by label) ─▶ top-3 past answers
                                                              │
                          hosted HF LLM  ──(fails)──▶ local ~1.5B LLM ─▶ draft reply
```

## Setup

Python 3.11 is targeted (development ran on 3.12).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # paste your HF_TOKEN
```

## Run order

```bash
python -m src.data                 # download + clean + split -> data/processed/
python -m src.baseline             # TF-IDF + LogReg baseline
python -m src.baseline zeroshot 200   # zero-shot BART-MNLI on a 200-ticket subsample (slow)
python -m src.finetune --epochs 3  # fine-tune DistilBERT -> models/ticket-classifier
python -m src.rag build            # build the ChromaDB index (train split only)
python -m src.evaluate             # comparison table -> results/comparison.md
pytest                             # unit tests
uvicorn src.api:app                # API docs at http://127.0.0.1:8000/docs
streamlit run app.py               # UI
```

## Results (test split, 2,428 tickets, from the runs in `results/`)

| Model | n | Accuracy | Macro F1 |
|---|---|---|---|
| Zero-shot BART-large-MNLI | 200 (stratified subsample) | 0.535 | 0.484 |
| TF-IDF + LogReg | 2,428 | 0.9967 | 0.9965 |
| Fine-tuned DistilBERT (3 epochs) | 2,428 | 0.9992 | 0.9991 |

Retrieval: the top-1 retrieved ticket has the same intent as the test ticket 99.4% of the time (500 sampled test tickets).
Details: `results/comparison.md`, `results/errors.csv`, `results/*_metrics.json`.

## Data

[bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (cdla-sharing-1.0). Its 27 intents are mapped to 4 triage labels in `src/data.py`. Duplicates and conflicting texts are removed before a stratified 80/10/10 split.

## Limitations (please read)

- **The data is synthetic and templated.** Near-perfect scores mostly show the task is easy on this data; even TF-IDF reaches 99.7%. Expect much lower accuracy on real, messy tickets. The 8 errors we see are typos ("tfack", "apyment").
- **No "bug / technical issue" label**, because the dataset has none. The `general` class is a catch-all, which is why zero-shot does poorly on it.
- **The zero-shot number is on only 200 tickets** (BART-large is slow on CPU), so it is less precise than the others.
- **Retrieved answers are templated.** Generated replies are only as good as them; the prompt tells the model to escalate when the references don't cover the question.
- **Reply quality has not been measured** (no automatic metric was run); inspect samples yourself.
- The hosted-model path needs `HF_TOKEN`; free-tier availability of the configured model can change, so the model name is the `GEN_MODEL` setting and a local fallback exists.

## Deploying to Hugging Face Spaces

1. Push the classifier: `python -m src.finetune --push <user>/ticket-classifier` (needs a write token).
2. Create a Streamlit Space and push `app.py`, `src/`, `requirements.txt`.
3. In Space settings, add secrets `HF_TOKEN` and the variable `CLASSIFIER_DIR=<user>/ticket-classifier`.
4. The vector index must exist on first start: run `python -m src.data && python -m src.rag build` at startup (the `Dockerfile` does this).

Docker: `docker build -t triage . && docker run -p 8501:8501 --env-file .env triage` (not yet built or tested in this environment).
