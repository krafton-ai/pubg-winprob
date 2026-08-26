"""
Evaluation metrics for PGC models.
"""

import os
from typing import Dict, List
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models import TransformerBackbone, SurvivalTimeHead
from src.data.dataset import collate_fn
from src.training.metrics import (
    compute_winner_accuracy,
    compute_c_index,
    compute_integrated_brier_score,
    compute_log_loss,
    compute_ece,
    compute_all_metrics,
)

# Constants
from src.data.utils import NUM_PHASES, time_point_to_phase
# Primary metrics (reported in the paper) first, followed by calibration diagnostics.
METRIC_NAMES = ['accuracy', 'c_index', 'ibs', 'ece', 'log_loss']


def evaluate_test_set(
    backbone: TransformerBackbone,
    head: SurvivalTimeHead,
    test_loader: DataLoader,
    device: str = "cpu",
    save_path: str = None,
) -> Dict:
    """
    Evaluate model on test set with phase-wise accuracy.

    Accuracy is computed as: whether the squad with highest predicted
    survival time matches the squad with highest actual survival time.

    Each match consists of phase-sampled time points (10 phases, non-uniform
    density); phase membership is derived from the time point value.

    Args:
        backbone: Trained backbone model.
        head: Trained prediction head.
        test_loader: Test DataLoader.
        device: Device to use.
        save_path: Path to save results CSV. If None, not saved.

    Returns:
        Dictionary with phase-wise accuracies and average.
    """
    backbone = backbone.to(device)
    head = head.to(device)
    backbone.eval()
    head.eval()

    # Collect predictions and targets per match_id
    match_data = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x_continuous"].to(device)
            positions = batch["positions"].to(device)
            bluezone = batch["bluezone_info"].to(device)
            whitezone = batch["whitezone_info"].to(device)
            map_idx = batch["map_idx"].to(device)
            y = batch["y"]
            time_points = batch["time_point"]

            # Forward pass
            embeddings = backbone(x, positions, bluezone, whitezone, map_idx)
            predictions = head(embeddings)

            # Move to CPU
            predictions = predictions.cpu()

            # Process each sample in batch
            batch_size = x.size(0)
            for i in range(batch_size):
                time_point = time_points[i].item()
                pred = predictions[i]  # (num_squads,)
                target = y[i]  # (num_squads,)

                match_data[time_point].append({
                    "time_point": time_point,
                    "predictions": pred,
                    "targets": target,
                })

    # Sort by time_point and compute phase-wise accuracy
    all_time_points = sorted(match_data.keys())

    phase_correct = defaultdict(int)
    phase_total = defaultdict(int)

    for time_point in all_time_points:
        # Determine phase (1-indexed) from the time point value
        phase = time_point_to_phase(time_point)

        for sample in match_data[time_point]:
            pred = sample["predictions"]
            target = sample["targets"]

            # Find squad with highest prediction and highest target
            pred_winner = pred.argmax().item()
            target_winner = target.argmax().item()

            # Check if correct
            if pred_winner == target_winner:
                phase_correct[phase] += 1
            phase_total[phase] += 1

    # Compute accuracies
    results = {}
    total_correct = 0
    total_samples = 0

    for phase in range(1, NUM_PHASES + 1):
        if phase_total[phase] > 0:
            acc = phase_correct[phase] / phase_total[phase]
        else:
            acc = 0.0
        results[f"Phase_{phase}"] = acc
        total_correct += phase_correct[phase]
        total_samples += phase_total[phase]

    # Average accuracy
    if total_samples > 0:
        results["Average"] = total_correct / total_samples
    else:
        results["Average"] = 0.0

    # Save to CSV if path provided
    if save_path is not None:
        df = pd.DataFrame([results])
        cols = [f"Phase_{i}" for i in range(1, NUM_PHASES + 1)] + ["Average"]
        df = df[cols]
        df.to_csv(save_path, index=False)
        print(f"Results saved to: {save_path}")

    return results


def evaluate_test_set_by_match(
    backbone: TransformerBackbone,
    head: SurvivalTimeHead,
    test_dataset,
    device: str = "cpu",
    batch_size: int = 32,
    save_path: str = None,
) -> Dict:
    """
    Evaluate model on test set with phase-wise metrics, grouped by match_id.

    For each match_id, time_points are sorted and divided into 10 phases
    (5 time points per phase, 50 total per match).

    Metrics computed (paper reports Accuracy, C-index, and IBS):
    - Accuracy: Whether predicted winner matches actual winner
    - C-index: Concordance index for survival-time ranking
    - IBS: Integrated Brier Score for probabilistic predictions
    - ECE: Expected Calibration Error (calibration diagnostic)
    - Log Loss: Cross-entropy for winner prediction (calibration diagnostic)

    Args:
        backbone: Trained backbone model.
        head: Trained prediction head.
        test_dataset: Test PGCDataset (not DataLoader).
        device: Device to use.
        batch_size: Batch size for inference.
        save_path: Path to save results CSV. If None, not saved.

    Returns:
        Dictionary with phase-wise metrics and averages.
    """
    backbone = backbone.to(device)
    head = head.to(device)
    backbone.eval()
    head.eval()

    # Step 1: Build phase mapping for each dataset index
    match_groups = defaultdict(list)
    for idx in range(len(test_dataset)):
        match_id, time_point = test_dataset.group_keys[idx]
        match_groups[match_id].append((time_point, idx))

    for match_id in match_groups:
        match_groups[match_id].sort(key=lambda x: x[0])

    idx_to_phase = {}
    for match_id, samples in match_groups.items():
        for time_point, dataset_idx in samples:
            idx_to_phase[dataset_idx] = time_point_to_phase(time_point)

    # Step 2: Batch inference and collect results per phase
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Store predictions and targets by phase
    phase_preds = defaultdict(list)
    phase_targets = defaultdict(list)
    phase_masks = defaultdict(list)

    current_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x_continuous"].to(device)
            positions = batch["positions"].to(device)
            bluezone = batch["bluezone_info"].to(device)
            whitezone = batch["whitezone_info"].to(device)
            map_idx = batch["map_idx"].to(device)
            y = batch["y"]
            alive_mask = batch["squad_alive_mask"]
            padding_mask = batch["padding_mask"]

            valid_mask = alive_mask & padding_mask

            # Forward pass
            embeddings = backbone(x, positions, bluezone, whitezone, map_idx)
            predictions = head(embeddings).cpu()

            # Store by phase
            batch_size_actual = x.size(0)
            for i in range(batch_size_actual):
                dataset_idx = current_idx + i
                phase = idx_to_phase[dataset_idx]

                phase_preds[phase].append(predictions[i:i+1])
                phase_targets[phase].append(y[i:i+1])
                phase_masks[phase].append(valid_mask[i:i+1])

            current_idx += batch_size_actual

    # Step 3: Compute metrics per phase
    results = {metric: {} for metric in METRIC_NAMES}

    for phase in range(1, NUM_PHASES + 1):
        if phase in phase_preds and len(phase_preds[phase]) > 0:
            preds = torch.cat(phase_preds[phase], dim=0)
            targets = torch.cat(phase_targets[phase], dim=0)
            masks = torch.cat(phase_masks[phase], dim=0)

            metrics = compute_all_metrics(preds, targets, masks)

            for metric_name in METRIC_NAMES:
                results[metric_name][f"Phase_{phase}"] = metrics[metric_name]
        else:
            for metric_name in METRIC_NAMES:
                results[metric_name][f"Phase_{phase}"] = 0.0

    # Compute average metrics across all phases
    all_preds = []
    all_targets = []
    all_masks = []
    for phase in range(1, NUM_PHASES + 1):
        if phase in phase_preds:
            all_preds.extend(phase_preds[phase])
            all_targets.extend(phase_targets[phase])
            all_masks.extend(phase_masks[phase])

    if all_preds:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        avg_metrics = compute_all_metrics(all_preds, all_targets, all_masks)
        for metric_name in METRIC_NAMES:
            results[metric_name]["Average"] = avg_metrics[metric_name]
    else:
        for metric_name in METRIC_NAMES:
            results[metric_name]["Average"] = 0.0

    # Save to CSV if path provided
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Save each metric as a separate CSV
        for metric_name in METRIC_NAMES:
            metric_results = results[metric_name]
            df = pd.DataFrame([metric_results])
            cols = [f"Phase_{i}" for i in range(1, NUM_PHASES + 1)] + ["Average"]
            df = df[cols]

            metric_path = save_path.replace(".csv", f"_{metric_name}.csv")
            df.to_csv(metric_path, index=False)

        # Also save a combined summary
        summary_data = []
        for metric_name in METRIC_NAMES:
            row = {"metric": metric_name}
            row.update(results[metric_name])
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(save_path, index=False)
        print(f"Results saved to: {save_path}")

    return results


def print_results(results: Dict, title: str = "Test Results"):
    """Print evaluation results in a formatted table."""
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    # Check if results contain multiple metrics
    if 'accuracy' in results:
        # New format with multiple metrics
        phases = [f"Phase_{i}" for i in range(1, NUM_PHASES + 1)]

        # Header
        header = f"{'Phase':<12}"
        for metric in METRIC_NAMES:
            header += f"{metric.upper():<15}"
        print(header)
        print("-" * 90)

        # Phase rows
        for phase in phases:
            row = f"{phase:<12}"
            for metric in METRIC_NAMES:
                val = results[metric].get(phase, 0.0)
                row += f"{val:<15.4f}"
            print(row)

        # Average row
        print("-" * 90)
        avg_row = f"{'Average':<12}"
        for metric in METRIC_NAMES:
            val = results[metric].get("Average", 0.0)
            avg_row += f"{val:<15.4f}"
        print(avg_row)

    else:
        # Old format (backward compatibility)
        phases = [k for k in results.keys() if k.startswith("Phase_")]
        phases.sort(key=lambda x: int(x.split("_")[1]))

        print(f"{'Phase':<15} {'Accuracy':<15}")
        print("-" * 30)

        for phase in phases:
            print(f"{phase:<15} {results[phase]:.4f}")

        print("-" * 30)
        print(f"{'Average':<15} {results['Average']:.4f}")

    print("=" * 90)


