#!/bin/bash

# ============================================
# PGC Model Inference Script
# ============================================
# Usage:
#   Single CSV:
#     bash scripts/run_inference.sh --checkpoint_dir <dir> --csv_path <path>
#
#   Multiple CSVs (Phase-wise accuracy):
#     bash scripts/run_inference.sh --checkpoint_dir <dir> --folder_path <dir> --file_list "f1.csv f2.csv"
# ============================================

set -e

# Default values
CHECKPOINT_DIR=""
CSV_PATH=""
FOLDER_PATH=""
FILE_LIST=""
OUTPUT_PATH=""
DEVICE="cpu"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --csv_path)
            CSV_PATH="$2"
            shift 2
            ;;
        --folder_path)
            FOLDER_PATH="$2"
            shift 2
            ;;
        --file_list)
            FILE_LIST="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash scripts/run_inference.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint_dir   Directory containing best.pt and config.json (required)"
            echo "  --csv_path         Path to single CSV file for inference"
            echo "  --folder_path      Path to folder containing CSV files"
            echo "  --file_list        Space-separated list of CSV filenames (quoted)"
            echo "  --output_path      Path to save results (optional)"
            echo "  --device           Device to use: cpu, cuda, cuda:0, etc. (default: cpu)"
            echo ""
            echo "Examples:"
            echo "  # Single CSV inference:"
            echo "  bash scripts/run_inference.sh \\"
            echo "      --checkpoint_dir checkpoints/20231126_exp \\"
            echo "      --csv_path data/test_match.csv \\"
            echo "      --device cuda"
            echo ""
            echo "  # Multiple CSVs with phase-wise accuracy:"
            echo "  bash scripts/run_inference.sh \\"
            echo "      --checkpoint_dir checkpoints/20231126_exp \\"
            echo "      --folder_path data \\"
            echo "      --file_list \"match1.csv match2.csv match3.csv\" \\"
            echo "      --device cuda"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "Error: --checkpoint_dir is required"
    exit 1
fi

# Check checkpoint files exist
CHECKPOINT_PATH="${CHECKPOINT_DIR}/best.pt"
CONFIG_PATH="${CHECKPOINT_DIR}/config.json"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Config not found: ${CONFIG_PATH}"
    exit 1
fi

echo "=========================================="
echo "PGC Model Inference"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Device: ${DEVICE}"
echo "=========================================="

# Build command
CMD="python -m src.training.inference"
CMD="${CMD} --checkpoint_path ${CHECKPOINT_PATH}"
CMD="${CMD} --device ${DEVICE}"

if [ -n "${CSV_PATH}" ]; then
    # Single CSV mode
    echo "Mode: Single CSV Inference"
    echo "CSV: ${CSV_PATH}"
    CMD="${CMD} --csv_path ${CSV_PATH}"
    
    if [ -n "${OUTPUT_PATH}" ]; then
        CMD="${CMD} --output_path ${OUTPUT_PATH}"
    fi
    
elif [ -n "${FOLDER_PATH}" ] && [ -n "${FILE_LIST}" ]; then
    # Multiple CSV mode
    echo "Mode: Multiple CSV Inference (Phase-wise Accuracy)"
    echo "Folder: ${FOLDER_PATH}"
    echo "Files: ${FILE_LIST}"
    echo "Phases: 10"
    
    CMD="${CMD} --folder_path ${FOLDER_PATH}"
    CMD="${CMD} --file_list ${FILE_LIST}"
    
    if [ -n "${OUTPUT_PATH}" ]; then
        CMD="${CMD} --output_path ${OUTPUT_PATH}"
    fi
    
else
    echo "Error: Either --csv_path or (--folder_path and --file_list) is required"
    exit 1
fi

echo ""
echo "Running: ${CMD}"
echo "=========================================="
echo ""

# Execute
eval ${CMD}

echo ""
echo "=========================================="
echo "Inference completed!"
echo "=========================================="


