#!/bin/bash
# Stage 1A: Train segmentation model (supervised only)
#
# This trains a segmentation model using only the 200 labeled cases.
# Use this as a baseline or when pseudo labels are not available.

set -e

DATA_ROOT="./data/train"
OUTPUT_DIR="./checkpoints/seg"
ENCODER="efficientnet-b4"
IMAGE_SIZE=512
BATCH_SIZE=8
NUM_EPOCHS=100
LR=1e-4

echo "========================================"
echo "Stage 1A: Segmentation Training"
echo "========================================"

python csv_train_seg.py \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --encoder ${ENCODER} \
    --encoder_weights imagenet \
    --num_seg_classes 3 \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LR} \
    --view both \
    --image_size ${IMAGE_SIZE} \
    --nsd_tolerance 2.0 \
    --scheduler cosine

echo ""
echo "Segmentation model saved to: ${OUTPUT_DIR}/best_model.pth"
