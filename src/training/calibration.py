"""
Temperature Scaling for model calibration.

This module implements temperature scaling to minimize ECE on the validation set.
Temperature scaling divides logits by a learned temperature parameter tau
before applying softmax, which improves probability calibration.
"""

import os
import json
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import PGCDataset, CONTINUOUS_FEATURES
from src.data.dataset import collate_fn
from src.data.dataset_v2 import (
    PGCDatasetV2,
    collate_fn as collate_fn_v2,
    COMPUTED_FEATURES,
)
from src.models import TransformerBackbone
from src.models.heads import get_head
from src.training.metrics import compute_ece


def collect_logits_and_targets(
    backbone: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect all logits and targets from a dataloader.

    Args:
        backbone: Backbone model.
        head: Head model.
        dataloader: DataLoader to collect from.
        device: Device to use.

    Returns:
        Tuple of (logits, targets, valid_masks) tensors.
    """
    backbone.eval()
    head.eval()

    all_logits = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x_continuous"].to(device)
            positions = batch["positions"].to(device)
            bluezone = batch["bluezone_info"].to(device)
            whitezone = batch["whitezone_info"].to(device)
            map_idx = batch["map_idx"].to(device)
            y = batch["y"]
            alive_mask = batch["squad_alive_mask"]
            padding_mask = batch["padding_mask"]

            valid_mask = alive_mask & padding_mask

            embeddings = backbone(x, positions, bluezone, whitezone, map_idx)
            logits = head(embeddings).cpu()

            all_logits.append(logits)
            all_targets.append(y)
            all_masks.append(valid_mask)

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_targets, dim=0),
        torch.cat(all_masks, dim=0),
    )


def compute_ece_with_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
    n_bins: int = 15,
) -> float:
    """
    Compute ECE with temperature-scaled logits.

    Args:
        logits: Raw model logits (batch, num_squads).
        targets: Ground truth targets (batch, num_squads).
        valid_mask: Valid mask (batch, num_squads).
        temperature: Temperature parameter (tau).
        n_bins: Number of confidence bins (paper: B = 15).

    Returns:
        ECE value.
    """
    scaled_logits = logits / temperature
    return compute_ece(scaled_logits, targets, valid_mask, n_bins=n_bins)


def find_optimal_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    tau_min: float = 0.1,
    tau_max: float = 2.0,
    tau_step: float = 0.01,
    n_bins: int = 15,
    refine: bool = True,
    verbose: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Find the temperature that minimizes validation ECE via grid search
    followed by local refinement.

    This function is model-agnostic: it only requires (logits, targets,
    valid_mask) arrays, so it can calibrate the Transformer as well as any
    baseline (pass log-probabilities as logits, e.g. from
    scripts/run_baselines.py).

    Defaults follow the paper: grid search over [0.1, 2.0] with step 0.01,
    followed by local refinement of the optimum, with B = 15 confidence bins.

    Args:
        logits: Raw model logits (batch, num_squads).
        targets: Ground truth targets (batch, num_squads).
        valid_mask: Valid mask (batch, num_squads).
        tau_min: Minimum temperature to search.
        tau_max: Maximum temperature to search.
        tau_step: Grid step size.
        n_bins: Number of confidence bins for ECE.
        refine: If True (default), refine the grid optimum with
            scipy.optimize.minimize_scalar.
        verbose: Whether to print progress.

    Returns:
        Tuple of (optimal_temperature, minimum_ece, tau_to_ece_dict).
    """
    def objective(tau):
        return compute_ece_with_temperature(logits, targets, valid_mask, tau, n_bins=n_bins)

    # Grid search: compute ECE for all tau values
    tau_values = np.arange(tau_min, tau_max + tau_step / 2, tau_step)
    tau_to_ece = {}
    optimal_tau, min_ece = tau_values[0], float("inf")
    for tau in tau_values:
        ece = objective(tau)
        tau_to_ece[f"{tau:.4f}"] = float(ece)
        if ece < min_ece:
            optimal_tau, min_ece = float(tau), float(ece)

    # Optional refinement around the grid optimum
    if refine:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            objective,
            bounds=(tau_min, tau_max),
            method="bounded",
            options={"xatol": 1e-4},
        )
        if result.fun < min_ece:
            optimal_tau, min_ece = float(result.x), float(result.fun)
            tau_to_ece[f"{optimal_tau:.4f}"] = min_ece

    if verbose:
        # Also compute ECE at tau=1.0 for comparison
        ece_no_scaling = compute_ece_with_temperature(logits, targets, valid_mask, 1.0, n_bins=n_bins)
        print(f"ECE without temperature scaling (tau=1.0): {ece_no_scaling:.6f}")
        print(f"Optimal temperature (tau): {optimal_tau:.4f}")
        print(f"ECE with optimal temperature: {min_ece:.6f}")
        print(f"ECE improvement: {(ece_no_scaling - min_ece) / ece_no_scaling * 100:.2f}%")

    return optimal_tau, min_ece, tau_to_ece


def calibrate_from_checkpoint(
    checkpoint_path: str,
    val_folder_path: str,
    val_file_list: list,
    config_path: str = None,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 4,
    save_tau: bool = True,
) -> Dict[str, Any]:
    """
    Load a model from checkpoint and find optimal temperature on validation set.

    Args:
        checkpoint_path: Path to model checkpoint (e.g., best.pt).
        val_folder_path: Path to folder containing validation data.
        val_file_list: List of validation file names.
        config_path: Path to config.json. If None, inferred from checkpoint_path.
        device: Device to use.
        batch_size: Batch size for inference.
        num_workers: Number of data loading workers.
        save_tau: Whether to save optimal tau to checkpoint directory.

    Returns:
        Dictionary with optimal_tau and ece values.
    """
    # Load config
    if config_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Check dataset version
    use_dataset_v2 = config.get("use_dataset_v2", True)

    # Create models with correct input dimension
    if use_dataset_v2:
        input_dim = config.get("input_dim", len(CONTINUOUS_FEATURES) + len(COMPUTED_FEATURES))
    else:
        input_dim = config.get("input_dim", len(CONTINUOUS_FEATURES))

    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    loss_type = config.get("loss_type", "mse")

    backbone = TransformerBackbone(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        encoder_type=config.get("encoder_type", "transformer"),
    )

    head = get_head(loss_type, embed_dim=embed_dim, dropout=dropout)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    head.load_state_dict(checkpoint["head_state_dict"])
    scaler_params = checkpoint.get("scaler_params", None)

    backbone = backbone.to(device)
    head = head.to(device)

    # Create validation dataset and loader with correct dataset version
    if use_dataset_v2:
        val_dataset = PGCDatasetV2(
            folder_path=val_folder_path,
            file_list=val_file_list,
            continuous_features=CONTINUOUS_FEATURES,
            scaler_params=scaler_params,
        )
        val_collate_fn = collate_fn_v2
    else:
        val_dataset = PGCDataset(
            folder_path=val_folder_path,
            file_list=val_file_list,
            continuous_features=CONTINUOUS_FEATURES,
            scaler_params=scaler_params,
        )
        val_collate_fn = collate_fn

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_collate_fn,
    )

    print(f"Collecting logits from {len(val_dataset)} validation samples...")

    # Collect logits and targets
    logits, targets, valid_mask = collect_logits_and_targets(
        backbone, head, val_loader, device
    )

    print("Finding optimal temperature...")

    # Find optimal temperature
    optimal_tau, min_ece, tau_to_ece = find_optimal_temperature(
        logits, targets, valid_mask, verbose=True
    )

    results = {
        "optimal_tau": optimal_tau,
        "ece_with_optimal_tau": min_ece,
        "ece_without_tau": compute_ece_with_temperature(
            logits, targets, valid_mask, 1.0
        ),
        "tau_to_ece": tau_to_ece,
    }

    # Save tau to checkpoint directory
    if save_tau:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        tau_path = os.path.join(checkpoint_dir, "temperature.json")
        with open(tau_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Temperature saved to: {tau_path}")

    return results


def calibrate_from_split_csv(
    checkpoint_path: str,
    folder_path: str,
    split_csv_path: str,
    config_path: str = None,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 4,
    save_tau: bool = True,
) -> Dict[str, Any]:
    """
    Load a model from checkpoint and find optimal temperature using split CSV.

    Args:
        checkpoint_path: Path to model checkpoint (e.g., best.pt).
        folder_path: Path to folder containing data files.
        split_csv_path: Path to split CSV file.
        config_path: Path to config.json. If None, inferred from checkpoint_path.
        device: Device to use.
        batch_size: Batch size for inference.
        num_workers: Number of data loading workers.
        save_tau: Whether to save optimal tau to checkpoint directory.

    Returns:
        Dictionary with optimal_tau and ece values.
    """
    import pandas as pd

    # Load split information
    split_df = pd.read_csv(split_csv_path)
    val_files = split_df[split_df["split"] == "val"]["filename"].tolist()

    print(f"Found {len(val_files)} validation files")

    return calibrate_from_checkpoint(
        checkpoint_path=checkpoint_path,
        val_folder_path=folder_path,
        val_file_list=val_files,
        config_path=config_path,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        save_tau=save_tau,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find optimal temperature for model calibration."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., checkpoints/exp/best.pt)",
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to folder containing data files",
    )
    parser.add_argument(
        "--split_csv_path",
        type=str,
        required=True,
        help="Path to split CSV file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    args = parser.parse_args()

    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = args.device

    results = calibrate_from_split_csv(
        checkpoint_path=args.checkpoint_path,
        folder_path=args.folder_path,
        split_csv_path=args.split_csv_path,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\n" + "=" * 50)
    print("Calibration Results")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.6f}")


