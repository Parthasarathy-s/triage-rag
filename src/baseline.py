"""Classical baseline: TF-IDF + logistic regression.

Run:  python -m src.baseline
Writes results/baseline_model.joblib and results/baseline_metrics.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from src.data import LABELS, SEED, load_split

RESULTS_DIR = Path("results")
C_GRID = [0.1, 1.0, 10.0, 100.0]  # inverse regularization strength; picked on val


def build_pipeline(c: float) -> Pipeline:
    """TF-IDF on word 1-2grams feeding a multinomial logistic regression.

    Bigrams capture short phrases ("track order"); sublinear_tf dampens repeated words.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=c, max_iter=1000, random_state=SEED)),
    ])


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict:
    """Accuracy, macro F1, per-class precision/recall/F1 and confusion matrix on `df`."""
    y_true, y_pred = df["label"], model.predict(df["text"])
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": {
            lab: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(support[i])}
            for i, lab in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": LABELS,  # rows = true label, columns = predicted label
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
    }


def select_c(train: pd.DataFrame, val: pd.DataFrame) -> tuple[float, dict[float, float]]:
    """Fit one model per C on train; return the C with the best val macro F1 and all scores."""
    scores = {}
    for c in C_GRID:
        model = build_pipeline(c).fit(train["text"], train["label"])
        scores[c] = f1_score(val["label"], model.predict(val["text"]), average="macro")
    return max(scores, key=scores.get), scores


def mean_latency_ms(model: Pipeline, texts: list[str]) -> float:
    """Average time to predict ONE ticket (single-item calls, as in a live API)."""
    start = time.perf_counter()
    for t in texts:
        model.predict([t])
    return (time.perf_counter() - start) / len(texts) * 1000


def main() -> None:
    train, val, test = (load_split(n) for n in ("train", "val", "test"))
    best_c, val_scores = select_c(train, val)
    model = build_pipeline(best_c).fit(train["text"], train["label"])

    # The test set is touched once, after all choices (C) were made on val.
    metrics = {
        "model": "tfidf_logreg",
        "best_C": best_c,
        "val_macro_f1_by_C": {str(c): s for c, s in val_scores.items()},
        "val": evaluate(model, val),
        "test": evaluate(model, test),
        "latency_ms_per_prediction": mean_latency_ms(model, test["text"].head(500).tolist()),
        "n_train": len(train),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, RESULTS_DIR / "baseline_model.joblib")
    (RESULTS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    t = metrics["test"]
    print(f"best C={best_c}  val macro F1 by C: {metrics['val_macro_f1_by_C']}")
    print(f"TEST accuracy={t['accuracy']:.4f}  macro_f1={t['macro_f1']:.4f}  "
          f"latency={metrics['latency_ms_per_prediction']:.2f} ms")
    print(pd.DataFrame(t["per_class"]).T.round(4))
    print(pd.DataFrame(t["confusion_matrix"]["matrix"], index=LABELS, columns=LABELS))




# ---------------------------------------------------------------------------
# Zero-shot baseline: no training, an NLI model scores "this text is about <label>".
# Run:  python -m src.baseline zeroshot [n_samples]
# ---------------------------------------------------------------------------
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
ZERO_SHOT_N = 300  # BART-large is slow on CPU, so we score a stratified subsample


def zero_shot_predict(texts: list[str]) -> list[str]:
    """Classify texts with the NLI zero-shot pipeline; returns the top label per text."""
    from transformers import pipeline  # lazy import: heavy and only needed here

    clf = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    hypothesis = "This customer support message is about {}."
    out = clf(texts, candidate_labels=LABELS, hypothesis_template=hypothesis, batch_size=16)
    return [o["labels"][0] for o in out]


def run_zero_shot(n: int = ZERO_SHOT_N) -> dict:
    """Evaluate zero-shot on a stratified test subsample and save metrics."""
    test = load_split("test")
    per = n // len(LABELS)  # equal count per label
    sample = pd.concat(
        [test[test["label"] == l].sample(per, random_state=SEED) for l in LABELS]
    ).reset_index(drop=True)
    start = time.perf_counter()
    preds = zero_shot_predict(sample["text"].tolist())
    ms = (time.perf_counter() - start) / len(sample) * 1000
    p, r, f, s = precision_recall_fscore_support(sample["label"], preds, labels=LABELS, zero_division=0)
    metrics = {
        "model": ZERO_SHOT_MODEL,
        "n_eval": len(sample),
        "accuracy": float(accuracy_score(sample["label"], preds)),
        "macro_f1": float(f1_score(sample["label"], preds, average="macro", labels=LABELS)),
        "per_class_f1": {l: float(x) for l, x in zip(LABELS, f)},
        "latency_ms_per_prediction": ms,
    }
    (RESULTS_DIR / "zeroshot_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "zeroshot":
        run_zero_shot(int(sys.argv[2]) if len(sys.argv) > 2 else ZERO_SHOT_N)
    else:
        main()
