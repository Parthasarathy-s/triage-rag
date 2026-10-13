"""Compare the three classifiers on the same held-out test set and analyse errors.

Run:  python -m src.evaluate
Needs: results/baseline_model.joblib (python -m src.baseline) and
       models/ticket-classifier (python -m src.finetune).
Zero-shot numbers are read from results/zeroshot_metrics.json (it uses a subsample).
Writes results/comparison.md and results/errors.csv.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data import LABELS, load_split
from src.finetune import MODEL_DIR, TicketClassifier

RESULTS_DIR = Path("results")


def scores(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Accuracy and macro F1 (macro = unweighted mean over classes)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS)),
    }


def error_table(test: pd.DataFrame, base_pred: list[str], ft_pred: list[str]) -> pd.DataFrame:
    """Rows where at least one model is wrong, for manual error analysis."""
    df = test.assign(baseline_pred=base_pred, finetuned_pred=ft_pred)
    wrong = (df["label"] != df["baseline_pred"]) | (df["label"] != df["finetuned_pred"])
    return df[wrong][["text", "intent", "label", "baseline_pred", "finetuned_pred"]]


def retrieval_hit_rate(test: pd.DataFrame, n: int = 500) -> dict[str, float]:
    """Does the top-1 retrieved ticket share the true intent / label? (unfiltered search)."""
    from src.rag import retrieve

    sample = test.sample(min(n, len(test)), random_state=42)
    hits = [retrieve(t, k=1)[0] for t in sample["text"]]
    return {
        "n": len(sample),
        "top1_same_intent": sum(h.intent == i for h, i in zip(hits, sample["intent"])) / len(sample),
        "top1_same_label": sum(h.label == l for h, l in zip(hits, sample["label"])) / len(sample),
    }


def main() -> None:
    test = load_split("test")
    y = test["label"].tolist()

    base = joblib.load(RESULTS_DIR / "baseline_model.joblib")
    base_pred = base.predict(test["text"]).tolist()

    clf = TicketClassifier(MODEL_DIR)
    start = time.perf_counter()
    ft_pred = [l for l, _ in clf.predict(test["text"].tolist())]
    ft_ms = (time.perf_counter() - start) / len(test) * 1000  # batched, so lower than live API latency

    rows = [
        {"model": "TF-IDF + LogReg", "n": len(y), **scores(y, base_pred)},
        {"model": "Fine-tuned DistilBERT", "n": len(y), **scores(y, ft_pred), "ms_per_ticket": ft_ms},
    ]
    zs_path = RESULTS_DIR / "zeroshot_metrics.json"
    if zs_path.exists():
        zs = json.loads(zs_path.read_text())
        rows.insert(0, {"model": "Zero-shot BART-MNLI (subsample)", "n": zs["n_eval"],
                        "accuracy": zs["accuracy"], "macro_f1": zs["macro_f1"],
                        "ms_per_ticket": zs["latency_ms_per_prediction"]})
    table = pd.DataFrame(rows)

    errors = error_table(test, base_pred, ft_pred)
    errors.to_csv(RESULTS_DIR / "errors.csv", index=False)
    cm = pd.DataFrame(confusion_matrix(y, ft_pred, labels=LABELS), index=LABELS, columns=LABELS)

    ret = retrieval_hit_rate(test)
    md = ["# Model comparison (test split)\n", "```\n" + table.round(4).to_string(index=False) + "\n```",
          "\n\n## Fine-tuned confusion matrix (rows = true, cols = predicted)\n", "```\n" + cm.to_string() + "\n```",
          f"\n\n{len(errors)} test tickets are misclassified by at least one model (see errors.csv).\n",
          f"\n## Retrieval (top-1, {ret['n']} test tickets)\n\nsame intent: {ret['top1_same_intent']:.3f}, "
          f"same label: {ret['top1_same_label']:.3f}\n"]
    (RESULTS_DIR / "comparison.md").write_text("\n".join(md))
    print("\n".join(md))
    print(errors.head(15).to_string())


if __name__ == "__main__":
    main()
