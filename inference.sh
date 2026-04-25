#!/bin/bash
# Run banking intent inference
#
# Usage:
#   bash inference.sh                           — use default example message
#   bash inference.sh "your custom message"     — use a custom message

MESSAGE="${1:-I lost my card and need a replacement}"

echo "=== Banking Intent Inference ==="
python scripts/inference.py \
    --config configs/inference.yaml \
    --input "$MESSAGE"

echo "=== Done ==="
