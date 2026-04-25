"""
preprocess_data.py
------------------
Load, sample, preprocess, and split the BANKING77 dataset.
Data is loaded directly from PolyAI's GitHub repo (CSV format) to
avoid the 'Dataset scripts are no longer supported' error in newer
versions of the HuggingFace datasets library.

Usage:
    python scripts/preprocess_data.py --config configs/train.yaml
"""

import os
import re
import json
import argparse
import random

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# Banking77 raw CSV hosted on PolyAI's GitHub (same data as HuggingFace)
_BANKING77_BASE = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/banking_data"
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """Basic text normalization."""
    text = text.lower().strip()
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing punctuation artifacts
    text = text.strip("\"'")
    return text


def load_banking77_from_github():
    """
    Load BANKING77 train/test splits directly from PolyAI's GitHub CSV.
    Returns two DataFrames with columns: 'text', 'category'
    """
    train_df = pd.read_csv(f"{_BANKING77_BASE}/train.csv")
    test_df  = pd.read_csv(f"{_BANKING77_BASE}/test.csv")
    return train_df, test_df


def sample_intents_from_df(train_df, num_intents: int):
    """
    Select top-K intents by frequency from the train DataFrame.
    Returns label mappings.
    """
    counter     = Counter(train_df["category"])
    top_intents = [cat for cat, _ in counter.most_common(num_intents)]

    print(f"Selected {len(top_intents)} intents out of {train_df['category'].nunique()}")
    print(f"Intents: {top_intents[:10]} ...")

    label2id = {lbl: i for i, lbl in enumerate(top_intents)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return set(top_intents), id2label, label2id


def filter_and_format_df(df, intent_set: set, label2id: dict, id2label: dict):
    """Filter a DataFrame to selected intents and return records list."""
    records = []
    for _, row in df.iterrows():
        if row["category"] in intent_set:
            label_id = label2id[row["category"]]
            records.append({
                "text":     normalize_text(row["text"]),
                "label_id": label_id,
            })
    return records


def build_system_prompt(intent_list: list) -> str:
    """Build the system prompt containing all valid intent labels."""
    intents_str = "\n".join(f"- {intent}" for intent in intent_list)
    return (
        f"You are a banking intent classifier. "
        f"Given a customer query, classify it into exactly one of the "
        f"following {len(intent_list)} intent categories:\n"
        f"{intents_str}\n\n"
        f"Respond with ONLY the intent name, nothing else."
    )


def build_sft_prompt(text: str, label_name: str, system_prompt: str) -> str:
    """Format a training example using Llama 3.1 ChatML format."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{label_name}<|eot_id|>"
    )


def main(config_path: str):
    config = load_config(config_path)

    # ── Config values ─────────────────────────────────────────────────────
    num_intents = config["data"]["num_intents"]
    val_size    = config["data"].get("val_size", 0.1)   # 10% of train → val
    test_size   = config["data"]["test_size"]
    seed        = config["data"]["random_seed"]
    out_dir     = "sample_data"
    train_out   = config["data"]["train_path"]
    val_out     = config["data"].get("val_path", "sample_data/val.csv")
    test_out    = config["data"]["test_path"]
    label_map_path = config["data"]["label_map_path"]

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    # ── 1. Load BANKING77 ─────────────────────────────────────────────────
    print("[1/5] Loading BANKING77 dataset from GitHub CSV...")
    raw_train_df, raw_test_df = load_banking77_from_github()
    print(f"  Train size: {len(raw_train_df)} | Test size: {len(raw_test_df)}")
    print(f"  Total intents: {raw_train_df['category'].nunique()}")

    # ── 2. Sample K intents ───────────────────────────────────────────────
    print(f"\n[2/5] Sampling top-{num_intents} intents by frequency...")
    intent_set, id2label, label2id = sample_intents_from_df(
        raw_train_df, num_intents=num_intents
    )
    # Build ordered intent list & system prompt (used in every example)
    intent_list   = [id2label[i] for i in range(len(id2label))]
    system_prompt = build_system_prompt(intent_list)

    # ── 3. Filter & format splits ──────────────────────────────────────────
    print("\n[3/5] Filtering and normalizing data...")
    train_records = filter_and_format_df(raw_train_df, intent_set, label2id, id2label)
    test_records  = filter_and_format_df(raw_test_df,  intent_set, label2id, id2label)
    print(f"  Filtered train: {len(train_records)} samples")
    print(f"  Filtered test : {len(test_records)} samples")

    # ── 4. Add label names and SFT prompts (ChatML format) ────────────────
    print("\n[4/5] Adding label names and ChatML-formatted prompts...")
    for rec in train_records:
        rec["label_name"] = id2label[rec["label_id"]]
        rec["prompt"]     = build_sft_prompt(rec["text"], rec["label_name"], system_prompt)

    for rec in test_records:
        rec["label_name"] = id2label[rec["label_id"]]
        rec["prompt"]     = build_sft_prompt(rec["text"], rec["label_name"], system_prompt)


    # ── 5. Split train → train + val (stratified) ─────────────────────────
    print(f"\n[5/6] Splitting train → train + validation (val_size={val_size})...")
    train_df = pd.DataFrame(train_records)
    test_df  = pd.DataFrame(test_records)

    # Stratified split giữ nguyên tỉ lệ class ở cả 2 tập
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df["label_id"],  # đảm bảo phân phối class đều nhau
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"  Final train : {len(train_df)} samples")
    print(f"  Validation  : {len(val_df)} samples")
    print(f"  Test        : {len(test_df)} samples")

    # ── 6. Save to CSV ────────────────────────────────────────────────────
    print(f"\n[6/6] Saving processed data...")
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out,     index=False)
    test_df.to_csv(test_out,   index=False)
    print(f"  train.csv → {train_out} ({len(train_df)} rows)")
    print(f"  val.csv   → {val_out}   ({len(val_df)} rows)")
    print(f"  test.csv  → {test_out}  ({len(test_df)} rows)")

    # Save label map
    label_map = {"id2label": id2label, "label2id": label2id}
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  label_map.json → {label_map_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n─── Summary ───────────────────────────────────")
    print(f"  Intents selected : {num_intents}")
    print(f"  Train samples    : {len(train_df)}")
    print(f"  Val samples      : {len(val_df)}")
    print(f"  Test samples     : {len(test_df)}")
    print(f"\nSample prompt:\n{train_df['prompt'].iloc[0]}")
    print("─────────────────────────────────────────────")
    print("✅ Preprocessing complete!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BANKING77 dataset")
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml",
        help="Path to the training config YAML file"
    )
    args = parser.parse_args()
    main(args.config)
