#!/bin/bash

MODEL_NAME="PixArt-alpha/PixArt-XL-2-512x512"
DATASET_NAME="rapidata"          # pickapic-anonymous/pickapic_v1 | rapidata | hpdv2 | roco (roco is --sft only)
TRAIN_DATA_DIR="."               # dir holding the local datasets built by create_*.py, this hyperparam needs to be omitted if using cache_dir for pick-a-pic for e.g.
CACHE_DIR="path/to/cache"

SEEDS=(42 43 44)
STRATEGY="eigen"                 # eigen | kcenter | facility_location | kmeans++ | semdedup (same budget)
# rapidata SFT settings
FIRST_EIGEN=1024
RATIO=6
BIN_COUNT=4
THRESHOLD=0.4
THRESH_STAGE2=0.6
MAX_STEPS=1792                   # kcenter, kmeans++ and semdedup on rapidata keep 4,888 rows, 1222 steps for the same epochs
WARMUP=150
BATCH_SIZE=4
BETA_DPO=5000
LR=1e-5

for seed in "${SEEDS[@]}"; do
    # runs resume from --output_dir, so use a separate OUTPUT per variant
    OUTPUT="${seed}_${FIRST_EIGEN}_${STRATEGY}_pixart_${DATASET_NAME//\//_}"
    accelerate launch --mixed_precision="fp16" train_scripts/train_pixart_dpo.py \
        --output_dir "$OUTPUT" \
        --seed="$seed" \
        --pretrained_model_name_or_path="$MODEL_NAME" \
        --dataset_name="$DATASET_NAME" \
        --train_data_dir="$TRAIN_DATA_DIR" \
        --cache_dir="$CACHE_DIR" \
        --train_batch_size="$BATCH_SIZE" \
        --dataloader_num_workers=1 \
        --gradient_accumulation_steps=1 \
        --max_train_steps="$MAX_STEPS" \
        --lr_scheduler="constant_with_warmup" \
        --lr_warmup_steps="$WARMUP" \
        --learning_rate="$LR" \
        --checkpointing_steps "$MAX_STEPS" \
        --beta_dpo "$BETA_DPO" \
        --first_eigen="$FIRST_EIGEN" \
        --ratio="$RATIO" \
        --bin_count="$BIN_COUNT" \
        --threshold="$THRESHOLD" \
        --thresh_stage2="$THRESH_STAGE2" \
        --strategy="$STRATEGY" \
        --validation_epochs 9999999 \
        --sft              # turn off for DPO
        # --random         # random-selection ablation
        # --full           # full-scale baseline (no selection)

    sleep 30
done