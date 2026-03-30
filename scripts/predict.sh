#!/bin/bash
# Final Prediction Pipeline
#
# Runs the full two-stage prediction on validation/test data:
#   Step 1: Segmentation model predicts masks
#   Step 2: 4-part classification model predicts risk class
#   Step 3: Merge segmentation + classification into submission format
#
# Output: data/val/preds.tar.gz (ready for submission)

set -e

VAL_DIR="./data/val"
SEG_CHECKPOINT="./checkpoints/seg_semi/best_model.pth"
CLS_CHECKPOINT="./checkpoints/cls_4parts/best_model.pth"

SEG_ENCODER="efficientnet-b4"
CLS_ENCODER="resnet152"
FUSION_METHOD="concat"
DILATION_KERNEL_SIZE=0
IMAGE_SIZE=512

echo "========================================"
echo "Two-Stage Prediction Pipeline"
echo "========================================"

# Check prerequisites
if [ ! -f "${SEG_CHECKPOINT}" ]; then
    echo "Error: Segmentation checkpoint not found: ${SEG_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${CLS_CHECKPOINT}" ]; then
    echo "Error: Classification checkpoint not found: ${CLS_CHECKPOINT}"
    exit 1
fi

if [ ! -d "${VAL_DIR}/images" ]; then
    echo "Error: Validation images not found: ${VAL_DIR}/images"
    exit 1
fi

# Step 1: Segmentation prediction
echo ""
echo "Step 1/3: Segmentation prediction"
echo "=================================="
python csv_predict_seg.py \
    --val-dir ${VAL_DIR} \
    --checkpoint ${SEG_CHECKPOINT} \
    --output-dir ${VAL_DIR}/preds_seg \
    --encoder ${SEG_ENCODER} \
    --num-seg-classes 3 \
    --view both \
    --resize-target ${IMAGE_SIZE}

# Step 2: 4-part classification prediction
echo ""
echo "Step 2/3: Classification prediction"
echo "===================================="
python csv_predict_cls_4parts.py \
    --val-dir ${VAL_DIR} \
    --checkpoint ${CLS_CHECKPOINT} \
    --output-dir ${VAL_DIR}/preds_cls \
    --masks-subdir preds_seg \
    --images-subdir images \
    --encoder ${CLS_ENCODER} \
    --fusion-method ${FUSION_METHOD} \
    --dilation-kernel-size ${DILATION_KERNEL_SIZE} \
    --num-cls-classes 2 \
    --device cuda

# Step 3: Merge predictions
echo ""
echo "Step 3/3: Merging predictions"
echo "=============================="
python csv_merge_predictions.py \
    --seg-dir ${VAL_DIR}/preds_seg \
    --cls-dir ${VAL_DIR}/preds_cls \
    --output-dir ${VAL_DIR}/preds \
    --create-archive

echo ""
echo "========================================"
echo "Prediction complete!"
echo "========================================"
echo "Submission file: ${VAL_DIR}/preds.tar.gz"
