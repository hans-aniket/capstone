"""
data/prepare_data.py
────────────────────
Downloads the amazon_polarity dataset from HuggingFace, creates a
stratified subsample, and saves train/test splits as Parquet files
under data/splits/ for use by all model training scripts.

Output:
    data/splits/train.parquet         (20 000 rows, balanced)
    data/splits/test.parquet          (5  000 rows, balanced)
    data/splits/splits_metadata.json  (reproducibility record)
"""

import json
import random
import pathlib
from datetime import datetime, timezone

import pandas as pd
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────
SEED           = 42
N_TRAIN_EACH   = 10_000   # per class -> 20k total
N_TEST_EACH    = 2_500    # per class ->  5k total
OUTPUT_DIR     = pathlib.Path(__file__).parent / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)


def subsample(split_data, n_per_class: int) -> list[dict]:
    """Return a balanced subsample from a HuggingFace dataset split."""
    pos = [row for row in split_data if row["label"] == 1]
    neg = [row for row in split_data if row["label"] == 0]
    sampled = random.sample(pos, n_per_class) + random.sample(neg, n_per_class)
    random.shuffle(sampled)
    return sampled


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert list of dicts to a clean DataFrame with combined text."""
    df = pd.DataFrame(rows)
    # Combine title + content into a single 'text' column
    df["text"] = df["title"].fillna("") + " " + df["content"].fillna("")
    df["text"] = df["text"].str.strip()
    return df[["text", "label"]]


def main():
    print("Loading amazon_polarity from HuggingFace … (this may take a moment)")
    dataset = load_dataset("amazon_polarity")

    print(f"Full dataset — train: {len(dataset['train']):,}  test: {len(dataset['test']):,}")

    print(f"Subsampling train -> {N_TRAIN_EACH * 2:,} rows …")
    train_rows = subsample(dataset["train"], N_TRAIN_EACH)

    print(f"Subsampling test  -> {N_TEST_EACH * 2:,} rows …")
    test_rows  = subsample(dataset["test"],  N_TEST_EACH)

    train_df = to_dataframe(train_rows)
    test_df  = to_dataframe(test_rows)

    train_path = OUTPUT_DIR / "train.parquet"
    test_path  = OUTPUT_DIR / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,  index=False)

    # ── Save reproducibility metadata ─────────────────────────────────
    meta = {
        "dataset":           "amazon_polarity",
        "seed":              SEED,
        "n_train_per_class": N_TRAIN_EACH,
        "n_test_per_class":  N_TEST_EACH,
        "train_total":       len(train_df),
        "test_total":        len(test_df),
        "label_map":         {"0": "negative", "1": "positive"},
        "text_fields":       ["title", "content"],
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }
    meta_path = OUTPUT_DIR / "splits_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Saved splits:")
    print(f"   {train_path}      ({len(train_df):,} rows)")
    print(f"   {test_path}       ({len(test_df):,} rows)")
    print(f"   {meta_path}")
    print(f"\nLabel distribution (train):\n{train_df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
