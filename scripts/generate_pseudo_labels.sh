#!/bin/bash
# (Optional) Generate pseudo labels for semi-supervised training
#
# Uses an ensemble of multiple segmentation models to vote on
# pixel-level labels for unlabeled data (cases 200-999).
# Requires multiple pre-trained segmentation model checkpoints.

set -e

IMAGES_DIR="./data/train/images"
OUTPUT_DIR="./data/pseudo_labels"
DEVICE="cuda"

echo "========================================"
echo "Pseudo Label Generation (Ensemble)"
echo "========================================"

python csv_semi_supervised_label.py \
    --images-dir ${IMAGES_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --start-idx 200 \
    --end-idx 1000 \
    --cls-value 0 \
    --device ${DEVICE}

echo ""
echo "Pseudo labels saved to: ${OUTPUT_DIR}"
