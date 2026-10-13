import pandas as pd

from src.data import INTENT_TO_LABEL, clean, normalize_text, split


def _raw(rows):
    return pd.DataFrame(rows, columns=["instruction", "intent"])


def test_normalize_text_collapses_whitespace_and_keeps_placeholders():
    assert normalize_text("  track  {{Order Number}} \n now ") == "track {{Order Number}} now"


def test_clean_dedupes_case_insensitively_and_drops_conflicts():
    raw = _raw([
        ("Track my order", "track_order"),
        ("track my order", "track_order"),        # duplicate
        ("help me", "get_refund"),                # same text, different labels
        ("help me", "recover_password"),
        ("reset password", "recover_password"),
    ])
    out = clean(raw)
    assert sorted(out["text"]) == ["Track my order", "reset password"]
    assert set(out["label"]) == {"shipping", "account"}


def test_clean_rejects_unmapped_intent():
    try:
        clean(_raw([("x", "not_an_intent")]))
    except ValueError as e:
        assert "not_an_intent" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_split_is_stratified_and_disjoint():
    labels = list(INTENT_TO_LABEL.values())
    df = pd.DataFrame({"text": [f"t{i}" for i in range(400)],
                       "label": [labels[i % len(labels)] for i in range(400)]})
    parts = split(df)
    assert sum(len(p) for p in parts.values()) == 400
    assert not set(parts["train"]["text"]) & set(parts["test"]["text"])
