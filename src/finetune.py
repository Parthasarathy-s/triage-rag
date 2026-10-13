"""Fine-tune DistilBERT for ticket classification.

Run:  python -m src.finetune            (full run, a few epochs)
      python -m src.finetune --epochs 1 --max-train 2000   (quick smoke run)
Saves the model + tokenizer to models/ticket-classifier and metrics to results/finetune_metrics.json.
Push to the Hub afterwards with:  python -m src.finetune --push <user>/<repo>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data import LABELS, SEED, load_split

BASE_MODEL = os.getenv("CLASSIFIER_BASE_MODEL", "distilbert-base-uncased")
MODEL_DIR = Path(os.getenv("CLASSIFIER_DIR", "models/ticket-classifier"))
RESULTS_DIR = Path("results")
MAX_LEN = 64  # tickets are short; truncating keeps training fast
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def to_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Tokenize a split into a HF Dataset with integer `labels`. Padding is done per batch."""
    ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": df["label"].map(LABEL2ID).tolist()})
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN), batched=True)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Metrics reported by the Trainer after each epoch."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}


def train(epochs: int, max_train: int | None, batch_size: int = 32, lr: float = 5e-5) -> dict:
    """Fine-tune on train, pick the best epoch on val, score test once, and save everything."""
    train_df, val_df, test_df = (load_split(n) for n in ("train", "val", "test"))
    if max_train:
        train_df = train_df.sample(min(max_train, len(train_df)), random_state=SEED)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "finetune_ckpt"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,  # keep the epoch with the best val macro F1
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        seed=SEED,
        report_to="none",
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=to_dataset(train_df, tok),
        eval_dataset=to_dataset(val_df, tok),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    val = trainer.evaluate()
    # Test set is used once, after the best epoch was chosen on val.
    test = trainer.predict(to_dataset(test_df, tok))
    test_metrics = compute_metrics((test.predictions, test.label_ids))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))
    metrics = {
        "base_model": BASE_MODEL,
        "epochs": epochs,
        "n_train": len(train_df),
        "val": {k: v for k, v in val.items() if k in ("eval_accuracy", "eval_macro_f1")},
        "test": test_metrics,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "finetune_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


class TicketClassifier:
    """Loads the saved fine-tuned model and predicts labels with confidence scores."""

    def __init__(self, model_dir: str | Path = MODEL_DIR) -> None:
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 64) -> list[tuple[str, float]]:
        """Return (label, probability) per text. no_grad: inference needs no gradients."""
        out: list[tuple[str, float]] = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i : i + batch_size], truncation=True, max_length=MAX_LEN,
                           padding=True, return_tensors="pt")
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            out += [(ID2LABEL[int(k)], float(c)) for k, c in zip(idx, conf)]
        return out


def push_to_hub(repo_id: str) -> None:
    """Upload the saved model to the HF Hub (needs HF_TOKEN with write access in .env)."""
    from dotenv import load_dotenv

    load_dotenv()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.push_to_hub(repo_id, token=os.environ["HF_TOKEN"])
    tok.push_to_hub(repo_id, token=os.environ["HF_TOKEN"])
    print(f"pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--push", metavar="REPO_ID", default=None)
    a = ap.parse_args()
    if a.push:
        push_to_hub(a.push)
    else:
        train(a.epochs, a.max_train)
