"""Data loading, cleaning and splitting for the ticket triage project.

Source: bitext/Bitext-customer-support-llm-chatbot-training-dataset (cdla-sharing-1.0).
The raw data has 11 categories / 27 intents. We map the *intent* column to our own
triage labels (see INTENT_TO_LABEL), then dedupe and make a stratified split.

Run as a script to build the splits:  python -m src.data
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SEED = 42  # fixed so splits are identical on every run
PROCESSED_DIR = Path("data/processed")

# Mapping from the dataset's fine-grained intents to our triage labels.
# The dataset has no bug/technical-issue content, so there is no "bug" label yet.
# To add one later, map intents here (or add a second dataset) - nothing else changes.
INTENT_TO_LABEL: dict[str, str] = {
    # billing: money, invoices, refunds, fees
    "check_invoice": "billing",
    "get_invoice": "billing",
    "check_payment_methods": "billing",
    "payment_issue": "billing",
    "check_refund_policy": "billing",
    "get_refund": "billing",
    "track_refund": "billing",
    "check_cancellation_fee": "billing",
    # account: profile, login, registration
    "create_account": "account",
    "delete_account": "account",
    "edit_account": "account",
    "recover_password": "account",
    "registration_problems": "account",
    "switch_account": "account",
    # shipping: delivery and where an order is
    "change_shipping_address": "shipping",
    "set_up_shipping_address": "shipping",
    "delivery_options": "shipping",
    "delivery_period": "shipping",
    "track_order": "shipping",
    # general: everything else
    "cancel_order": "general",
    "change_order": "general",
    "place_order": "general",
    "contact_customer_service": "general",
    "contact_human_agent": "general",
    "complaint": "general",
    "review": "general",
    "newsletter_subscription": "general",
}
LABELS: list[str] = sorted(set(INTENT_TO_LABEL.values()))


def load_raw() -> pd.DataFrame:
    """Download (or read from the HF cache) the raw dataset as a DataFrame."""
    from datasets import load_dataset  # imported here so tests of other funcs stay light

    return load_dataset(DATASET_ID, split="train").to_pandas()


def normalize_text(text: str) -> str:
    """Trim and collapse whitespace. Placeholders like {{Order Number}} are kept as-is."""
    return re.sub(r"\s+", " ", text).strip()


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with columns [text, intent, label], deduplicated.

    Duplicates matter: identical texts in train and test would inflate every metric.
    We dedupe on lower-cased text, and drop texts that appear under conflicting labels.
    """
    df = raw.rename(columns={"instruction": "text"})[["text", "intent"]].copy()
    df["text"] = df["text"].map(normalize_text)
    df = df[df["text"].str.len() > 0]
    df["label"] = df["intent"].map(INTENT_TO_LABEL)
    unmapped = df.loc[df["label"].isna(), "intent"].unique()
    if len(unmapped):
        raise ValueError(f"Intents missing from INTENT_TO_LABEL: {list(unmapped)}")

    key = df["text"].str.lower()
    # Same text with different labels is ambiguous: remove every copy.
    conflicting = df.groupby(key)["label"].transform("nunique") > 1
    df = df[~conflicting].copy()
    df = df[~df["text"].str.lower().duplicated(keep="first")]
    return df.reset_index(drop=True)


def split(
    df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split on `label` (keeps class ratios in every split)."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    # val_size is a fraction of the *whole* data, so rescale for the second split.
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["label"],
        random_state=seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path = PROCESSED_DIR) -> None:
    """Write each split to CSV so later phases don't need to re-download."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)


def load_split(name: str, in_dir: str | Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load one saved split ('train', 'val' or 'test')."""
    return pd.read_csv(Path(in_dir) / f"{name}.csv")


def main() -> None:
    raw = load_raw()
    df = clean(raw)
    splits = split(df)
    save_splits(splits)
    print(f"raw rows: {len(raw)}  after cleaning: {len(df)}")
    for name, frame in splits.items():
        print(f"{name}: {len(frame)} rows  {frame['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
