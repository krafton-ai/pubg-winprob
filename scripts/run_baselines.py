"""
Run Table 1 baseline models with the shared evaluation protocol.

Baselines: RandomGuess, RuleBased, LogReg, LogHaz, RFClf, LGBMReg, LGBMClf,
CoxPH, DeepSurv.

All models are evaluated identically to the Transformer: predictions are
grouped by (match_id, time_point) and scored with
``src.training.metrics.compute_all_metrics`` (Accuracy, C-index, IBS, ECE,
log loss). Winner probabilities are converted to scores via log(p) so that
the masked softmax inside the metrics recovers the model's probabilities.

Hyperparameter search grids per baseline are documented in the docstrings of
``src/baselines.py``; the CLI exposes each baseline's key hyperparameters so
the grids can be reproduced (e.g. loop over --coxph_alpha, --lgbm_num_leaves).

Usage:
    python scripts/run_baselines.py \
        --folder_path /path/to/data \
        --split_csv_path /path/to/data/split_files.csv \
        --models RandomGuess,RuleBased,LogReg,LogHaz,RFClf,LGBMReg,LGBMClf,CoxPH,DeepSurv \
        --output_dir results/baselines --seed 42
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.continuous_features import CONTINUOUS_FEATURES
from src.training.metrics import compute_all_metrics
from src.baselines import (
    predict_winner_probs,
    rule_based_probs,
    set_seed,
    train_coxph_baseline,
    train_deepsurv_baseline,
    train_lgbm_classifier,
    train_lgbm_regression,
    train_logreg,
    train_loghaz_baseline,
    train_rf_classifier,
)

ALL_MODELS = ['RandomGuess', 'RuleBased', 'LogReg', 'LogHaz', 'RFClf',
              'LGBMReg', 'LGBMClf', 'CoxPH', 'DeepSurv']
MAX_SQUADS = 16
LOG_EPS = 1e-12


def load_split_data(folder_path: str, split_csv_path: str):
    """Load train/val/test DataFrames using the shared split CSV."""
    split_df = pd.read_csv(split_csv_path)

    def load_files(split_name: str) -> pd.DataFrame:
        files = split_df[split_df['split'] == split_name]['filename'].tolist()
        dfs = []
        for f in tqdm(sorted(files), desc=f"Loading {split_name}"):
            path = os.path.join(folder_path, f)
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    return load_files('train'), load_files('val'), load_files('test')


def build_group_tensors(
    model,
    model_name: str,
    df: pd.DataFrame,
    seed: int,
    desc: str = "Predicting",
):
    """
    Run one baseline over all (match_id, time_point) groups and return
    (pred, target, valid_mask) tensors in the shared metric format.

    Predicted winner probabilities are stored as log(p) so that the masked
    softmax inside the metrics recovers p exactly; dividing log(p) by a
    temperature tau likewise implements temperature scaling.
    """
    preds, targets, masks = [], [], []

    for (match_id, time_point), group in tqdm(
        df.groupby(['match_id', 'time_point']),
        desc=f"{desc} {model_name}",
    ):
        group = group.sort_values('squad_number')
        alive_mask = (group['squad_alive_count'].values > 0)
        if alive_mask.sum() < 1:
            continue

        if model_name == 'RandomGuess':
            probs = predict_winner_probs(None, model_name,
                                         np.zeros((len(group), 1)), alive_mask,
                                         time_point=time_point, match_id=match_id,
                                         seed=seed)
        elif model_name == 'RuleBased':
            probs = rule_based_probs(group, alive_mask)
        else:
            X_group = np.nan_to_num(
                group[model.feature_cols_].values.astype(float), nan=0.0
            )
            probs = predict_winner_probs(model, model_name, X_group, alive_mask,
                                         time_point=time_point, match_id=match_id,
                                         seed=seed)

        n = len(group)
        pred_row = np.full(MAX_SQUADS, -np.inf)
        target_row = np.zeros(MAX_SQUADS)
        mask_row = np.zeros(MAX_SQUADS, dtype=bool)

        # log(p): the masked softmax inside the metrics recovers p exactly.
        pred_row[:n] = np.log(probs + LOG_EPS)
        target_row[:n] = group['squad_death_time'].values
        mask_row[:n] = alive_mask

        preds.append(pred_row)
        targets.append(target_row)
        masks.append(mask_row)

    pred_t = torch.tensor(np.array(preds), dtype=torch.float32)
    target_t = torch.tensor(np.array(targets), dtype=torch.float32)
    mask_t = torch.tensor(np.array(masks), dtype=torch.bool)

    return pred_t, target_t, mask_t


def evaluate_model(
    model,
    model_name: str,
    test_df: pd.DataFrame,
    seed: int,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluate one baseline over all (match_id, time_point) groups in the test
    set using ``compute_all_metrics``, optionally with temperature scaling.
    """
    pred_t, target_t, mask_t = build_group_tensors(
        model, model_name, test_df, seed, desc="Evaluating"
    )
    return compute_all_metrics(pred_t / temperature, target_t, mask_t)


def main():
    parser = argparse.ArgumentParser(description="Table 1 baseline runner")
    parser.add_argument("--folder_path", type=str, required=True,
                        help="Path to data folder")
    parser.add_argument("--split_csv_path", type=str, required=True,
                        help="Path to split CSV")
    parser.add_argument("--output_dir", type=str, default="results/baselines",
                        help="Output directory (not committed)")
    parser.add_argument("--models", type=str, default=",".join(ALL_MODELS),
                        help="Comma-separated list of baselines to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    # Per-baseline hyperparameters (see src/baselines.py for search grids)
    parser.add_argument("--coxph_alpha", type=float, default=10.0)
    parser.add_argument("--logreg_c", type=float, default=1.0)
    parser.add_argument("--rf_n_estimators", type=int, default=100)
    parser.add_argument("--rf_max_depth", type=int, default=10)
    parser.add_argument("--lgbm_num_leaves", type=int, default=31)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.1)
    parser.add_argument("--lgbm_n_estimators", type=int, default=100)
    parser.add_argument("--nn_epochs", type=int, default=20)
    parser.add_argument("--nn_batch_size", type=int, default=512)
    parser.add_argument("--nn_lr", type=float, default=1e-3)
    parser.add_argument("--loghaz_num_durations", type=int, default=50)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibrate", action="store_true",
                        help="Per-model temperature scaling: find tau* on the "
                             "validation set (grid [0.1, 2.0] step 0.01 with "
                             "local refinement, B=15 bins) before evaluating "
                             "on test")
    args = parser.parse_args()

    set_seed(args.seed)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_names if m not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}. Choose from {ALL_MODELS}")

    train_df, val_df, test_df = load_split_data(args.folder_path, args.split_csv_path)
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")

    feature_cols = list(CONTINUOUS_FEATURES)

    trainers = {
        'RandomGuess': lambda: None,
        'RuleBased': lambda: None,
        'LogReg': lambda: train_logreg(train_df, feature_cols, C=args.logreg_c,
                                       seed=args.seed),
        'LogHaz': lambda: train_loghaz_baseline(
            train_df, feature_cols, num_epochs=args.nn_epochs,
            batch_size=args.nn_batch_size, lr=args.nn_lr,
            num_durations=args.loghaz_num_durations, seed=args.seed),
        'RFClf': lambda: train_rf_classifier(
            train_df, feature_cols, n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth, seed=args.seed),
        'LGBMReg': lambda: train_lgbm_regression(
            train_df, feature_cols, num_leaves=args.lgbm_num_leaves,
            learning_rate=args.lgbm_learning_rate,
            n_estimators=args.lgbm_n_estimators, seed=args.seed),
        'LGBMClf': lambda: train_lgbm_classifier(
            train_df, feature_cols, num_leaves=args.lgbm_num_leaves,
            learning_rate=args.lgbm_learning_rate,
            n_estimators=args.lgbm_n_estimators, seed=args.seed),
        'CoxPH': lambda: train_coxph_baseline(
            train_df, feature_cols, alpha=args.coxph_alpha, seed=args.seed),
        'DeepSurv': lambda: train_deepsurv_baseline(
            train_df, feature_cols, val_df=val_df, num_epochs=args.nn_epochs,
            batch_size=args.nn_batch_size, lr=args.nn_lr, device=args.device,
            seed=args.seed),
    }

    all_results: List[Dict] = []
    for name in model_names:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        model = trainers[name]()
        if model is None and name not in ('RandomGuess', 'RuleBased'):
            print(f"  [SKIP] {name}: training failed or dependency missing")
            continue

        temperature = 1.0
        if args.calibrate:
            from src.training.calibration import find_optimal_temperature

            val_pred, val_target, val_mask = build_group_tensors(
                model, name, val_df, seed=args.seed, desc="Calibrating"
            )
            temperature, _, _ = find_optimal_temperature(
                val_pred, val_target, val_mask, verbose=True
            )

        metrics = evaluate_model(model, name, test_df, seed=args.seed,
                                 temperature=temperature)
        metrics['model'] = name
        if args.calibrate:
            metrics['temperature'] = temperature
        all_results.append(metrics)
        print("  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                               if k != 'model'))

    if all_results:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "baseline_results.csv")
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
