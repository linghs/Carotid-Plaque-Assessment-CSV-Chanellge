#!/bin/bash
# Stage 2: Generate segmentation predictions on training data
#
# The segmentation model predicts masks for the labeled training data (0-199).
# These predicted masks are then used to train the classification model,
# ensuring consistency between training and inference distributions.

set -e

DATA_ROOT="./data/train"
SEG_CHECKPOINT="./checkpoints/seg_semi/best_model.pth"
OUTPUT_DIR="./checkpoints/seg_semi/preds"
ENCODER="efficientnet-b4"
IMAGE_SIZE=512

echo "========================================"
echo "Stage 2: Generate Seg Predictions"
echo "========================================"

if [ ! -f "${SEG_CHECKPOINT}" ]; then
    echo "Error: Segmentation checkpoint not found: ${SEG_CHECKPOINT}"
    echo "Please train the segmentation model first (scripts/train_seg_semi.sh)"
    exit 1
fi

python csv_predict_train_data.py \
    --val-dir ${DATA_ROOT} \
    --checkpoint ${SEG_CHECKPOINT} \
    --output-dir ${OUTPUT_DIR} \
    --encoder ${ENCODER} \
    --num-seg-classes 3 \
    --num-cls-classes 2 \
    --use-classification \
    --view both \
    --resize-target ${IMAGE_SIZE} \
    --device cuda \
    --batch-size 1

echo ""
echo "Segmentation predictions saved to: ${OUTPUT_DIR}"
