#!/bin/bash
# Usage:
#   bash train.sh           — fresh training
#   bash train.sh --resume  — resume from latest checkpoint

RESUME_FLAG=""
if [[ "$1" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    echo "=== RESUME MODE: continuing from latest checkpoint ==="
fi

# Step 1: Preprocess data (safe to run again — idempotent)
echo "=== Preprocessing data ==="
python scripts/preprocess_data.py --config configs/train.yaml

# Step 2: Train model
echo "=== Training model ==="
python scripts/train.py --config configs/train.yaml $RESUME_FLAG

echo "=== Done ==="
