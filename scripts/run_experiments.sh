#!/bin/bash

# Hyperparameter Grid Search for PGC Transformer Model
# Usage: bash scripts/run_experiments.sh
#
# Data/output locations can be overridden via environment variables:
#   FOLDER_PATH, SPLIT_CSV_PATH, LOG_DIR, BASE_CHECKPOINT_DIR

# Data paths (replace with your own paths or export as environment variables)
FOLDER_PATH="${FOLDER_PATH:-/path/to/data}"
SPLIT_CSV_PATH="${SPLIT_CSV_PATH:-/path/to/data/split_files.csv}"
LOG_DIR="${LOG_DIR:-logs}"

# Toy mode: set to true to use only a small fraction of data for quick testing
TOY_MODE=false
if [ "$TOY_MODE" = true ]; then
    TOY_RATIO=0.03
else
    TOY_RATIO=1.0
fi

# Training settings
NUM_EPOCHS=10
PATIENCE=4
BATCH_SIZE=512
NUM_WORKERS=4
WORLD_SIZE=8  # Number of GPUs
SEED=42

# Loss type: 'mse' (regression), 'cox' (survival), 'ce' (classification),
# 'weighted_cox' (paper loss: elimination + survival with early-focus weighting)
LOSS_TYPE="weighted_cox"

# Dataset version: true for v2 (with distance features), false for v1
USE_DATASET_V2="true"

# Hyperparameter grid (includes the paper configuration: 64/8/8, dropout 0.1, lr 1e-4)
EMBED_DIMS=(64 128 256)
NUM_HEADS=(4 8)
NUM_LAYERS=(4 6 8)
DROPOUTS=(0.1 0.2)
LRS=(1e-3 1e-4)

# Base checkpoint directory (not committed; see .gitignore)
BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-checkpoints}"

echo "=========================================="
echo "Starting Hyperparameter Grid Search"
echo "=========================================="

# Create log directory
mkdir -p ${LOG_DIR}

# Loop over hyperparameters
for EMBED_DIM in "${EMBED_DIMS[@]}"; do
    for NUM_HEAD in "${NUM_HEADS[@]}"; do
        for NUM_LAYER in "${NUM_LAYERS[@]}"; do
            for DROPOUT in "${DROPOUTS[@]}"; do
                for LR in "${LRS[@]}"; do

                    # Create experiment name from hyperparameters
                    EXP_NAME="emb${EMBED_DIM}_head${NUM_HEAD}_layer${NUM_LAYER}_drop${DROPOUT}_lr${LR}_${LOSS_TYPE}"
                    CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/${EXP_NAME}"

                    echo ""
                    echo "=========================================="
                    echo "Running experiment: ${EXP_NAME}"
                    echo "=========================================="

                    # Run training
                    python -m src.training.trainer \
                        --folder_path ${FOLDER_PATH} \
                        --split_csv_path ${SPLIT_CSV_PATH} \
                        --checkpoint_dir ${CHECKPOINT_DIR} \
                        --embed_dim ${EMBED_DIM} \
                        --num_heads ${NUM_HEAD} \
                        --num_layers ${NUM_LAYER} \
                        --dropout ${DROPOUT} \
                        --lr ${LR} \
                        --batch_size ${BATCH_SIZE} \
                        --num_epochs ${NUM_EPOCHS} \
                        --patience ${PATIENCE} \
                        --num_workers ${NUM_WORKERS} \
                        --world_size ${WORLD_SIZE} \
                        --toy_ratio ${TOY_RATIO} \
                        --loss_type ${LOSS_TYPE} \
                        --use_dataset_v2 ${USE_DATASET_V2} \
                        --seed ${SEED} \
                        2>&1 | tee "${LOG_DIR}/${EXP_NAME}.log"
                    echo "Experiment ${EXP_NAME} completed."
                    echo "Checkpoint saved to: ${CHECKPOINT_DIR}"

                done
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: ${BASE_CHECKPOINT_DIR}"
echo "Logs saved in: ${LOG_DIR}"
echo "=========================================="

# Generate summary (written to the checkpoint directory; not committed)
echo ""
echo "Generating summary..."
python -c "
import os
import json
import glob

base_dir = '${BASE_CHECKPOINT_DIR}'
results = []

for exp_dir in sorted(glob.glob(os.path.join(base_dir, '*'))):
    if not os.path.isdir(exp_dir):
        continue

    exp_name = os.path.basename(exp_dir)
    history_path = os.path.join(exp_dir, 'history.json')
    config_path = os.path.join(exp_dir, 'config.json')

    if os.path.exists(history_path) and os.path.exists(config_path):
        with open(history_path) as f:
            history = json.load(f)
        with open(config_path) as f:
            config = json.load(f)

        results.append({
            'exp_name': exp_name,
            'best_val_loss': history['best_val_loss'],
            'best_epoch': history['best_epoch'],
            'embed_dim': config['embed_dim'],
            'num_heads': config['num_heads'],
            'num_layers': config['num_layers'],
            'dropout': config['dropout'],
            'lr': config['lr'],
        })

# Sort by best_val_loss
results = sorted(results, key=lambda x: x['best_val_loss'])

# Save summary to JSON
summary_path = os.path.join(base_dir, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Summary saved to: {summary_path}')
"
