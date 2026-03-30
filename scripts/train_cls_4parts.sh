#!/bin/bash
# Stage 3: Train 4-part classification model
#
# Uses segmentation-predicted masks to create 4 masked image parts:
#   1. long_img * mask_128 (longitudinal view, plaque region)
#   2. long_img * mask_255 (longitudinal view, vessel region)
#   3. trans_img * mask_128 (transverse view, plaque region)
#   4. trans_img * mask_255 (transverse view, vessel region)
#
# Each part goes through its own encoder, and features are fused for classification.

set -e

DATA_ROOT="./data/train"
SEG_PRED_DIR="./checkpoints/seg_semi/preds"
OUTPUT_DIR="./checkpoints/cls_4parts"
ENCODER="resnet152"
FUSION_METHOD="concat"
DILATION_KERNEL_SIZE=0
BATCH_SIZE=4
NUM_EPOCHS=50
IMAGE_SIZE=512

echo "========================================"
echo "Stage 3: 4-Part Classification Training"
echo "========================================"

if [ ! -d "${SEG_PRED_DIR}" ]; then
    echo "Error: Segmentation predictions not found: ${SEG_PRED_DIR}"
    echo "Please run scripts/generate_seg_preds.sh first"
    exit 1
fi

python csv_train_cls_from_seg_4parts.py \
    --data_root ${DATA_ROOT} \
    --seg_pred_dir ${SEG_PRED_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --encoder ${ENCODER} \
    --encoder_weights imagenet \
    --fusion_method ${FUSION_METHOD} \
    --dilation_kernel_size ${DILATION_KERNEL_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --image_size ${IMAGE_SIZE} \
    --learning_rate 1e-4 \
    --scheduler cosine \
    --seed 42

echo ""
echo "4-part classification model saved to: ${OUTPUT_DIR}/best_model.pth"
