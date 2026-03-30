#!/bin/bash
# Full Training Pipeline (end-to-end)
#
# Runs the complete training pipeline in order:
#   1. Train segmentation model (semi-supervised)
#   2. Generate segmentation predictions on training data
#   3. Train 4-part classification model
#
# Prerequisites:
#   - Data in ./data/train/ (images/, labels/)
#   - (Optional) Pseudo labels in ./data/pseudo_labels/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "Full Training Pipeline"
echo "========================================"
echo ""

echo "[1/3] Training segmentation model..."
bash "${SCRIPT_DIR}/train_seg_semi.sh"

echo ""
echo "[2/3] Generating segmentation predictions..."
bash "${SCRIPT_DIR}/generate_seg_preds.sh"

echo ""
echo "[3/3] Training 4-part classification model..."
bash "${SCRIPT_DIR}/train_cls_4parts.sh"

echo ""
echo "========================================"
echo "Full training pipeline complete!"
echo "========================================"
echo ""
echo "Checkpoints:"
echo "  Segmentation:    ./checkpoints/seg_semi/best_model.pth"
echo "  Classification:  ./checkpoints/cls_4parts/best_model.pth"
echo ""
echo "To run prediction:"
echo "  bash scripts/predict.sh"
