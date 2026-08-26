"""
Batch inference on arbitrary CSV files using a trained checkpoint.

This module is the standard inference entry point: given a CSV file
(or folder of CSV files) and a checkpoint, it runs the model and
returns/saves per-time-step winner probabilities.

For a single-match, demo-oriented variant that also exports detailed
per-squad state (health, positions, inventory) alongside predictions
in JSON form, use ``src.training.inference_with_json`` instead.
"""

import os
import json
from typing import Dict, List, Optional

import pandas as pd
import torch

from src.data import PGCDataset, CONTINUOUS_FEATURES
from src.data.dataset import TARGET_SCALE
from src.models import TransformerBackbone
from src.models.heads import get_head
from src.training.evaluation import (
    evaluate_test_set_by_match,
    print_results,
)


def load_temperature(checkpoint_path: str) -> Optional[float]:
    """
    Load temperature parameter from checkpoint directory if available.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., best.pt).

    Returns:
        Temperature value if found, None otherwise.
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    tau_path = os.path.join(checkpoint_dir, "temperature.json")

    if os.path.exists(tau_path):
        with open(tau_path, "r") as f:
            tau_data = json.load(f)
        return tau_data.get("optimal_tau", None)

    return None


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str = None,
    device: str = "cpu",
) -> tuple:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., best.pt).
        config_path: Path to config.json. If None, inferred from checkpoint_path.
        device: Device to load model on.

    Returns:
        Tuple of (backbone, head, scaler_params, config).
    """
    # Load config
    if config_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Create models
    input_dim = config.get("input_dim", len(CONTINUOUS_FEATURES))
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]

    backbone = TransformerBackbone(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        encoder_type=config.get("encoder_type", "transformer"),
    )

    # Get appropriate head based on loss type from config
    loss_type = config.get("loss_type", "mse")
    head = get_head(loss_type, embed_dim=embed_dim, dropout=dropout)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    head.load_state_dict(checkpoint["head_state_dict"])
    scaler_params = checkpoint.get("scaler_params", None)

    backbone = backbone.to(device)
    head = head.to(device)

    return backbone, head, scaler_params, config


def run_inference_on_csv(
    csv_path: str,
    checkpoint_path: str,
    config_path: str = None,
    device: str = "cpu",
    save_predictions: bool = True,
    output_path: str = None,
    temperature: float = None,
) -> pd.DataFrame:
    """
    Run inference on a single CSV file and return win probabilities.

    The model outputs raw scores (log hazard values), which are converted
    to win probabilities using softmax over valid (alive) squads.
    Temperature scaling can be applied for better calibration.

    Args:
        csv_path: Path to input CSV file.
        checkpoint_path: Path to model checkpoint.
        config_path: Path to config.json. If None, inferred.
        device: Device to use.
        save_predictions: Whether to save predictions to CSV.
        output_path: Output path for predictions. If None, auto-generated.
        temperature: Temperature for scaling logits before softmax.
                     If None, attempts to load from temperature.json in checkpoint dir.
                     If not found, uses 1.0 (no scaling).

    Returns:
        DataFrame with win probabilities for each squad.
    """
    # Load model
    backbone, head, scaler_params, config = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )

    if scaler_params is None:
        raise ValueError(
            "Checkpoint does not contain scaler_params. "
            "Please use a checkpoint saved with scaler_params."
        )

    # Load temperature if not provided
    if temperature is None:
        temperature = load_temperature(checkpoint_path)
        if temperature is not None:
            print(f"Using temperature from checkpoint: {temperature:.4f}")
        else:
            temperature = 1.0

    # Create dataset for single CSV
    folder_path = os.path.dirname(csv_path)
    file_name = os.path.basename(csv_path)

    dataset = PGCDataset(
        folder_path=folder_path,
        file_list=[file_name],
        continuous_features=CONTINUOUS_FEATURES,
        scaler_params=scaler_params,
    )

    backbone.eval()
    head.eval()

    results = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            match_id, time_point = dataset.group_keys[idx]
            sample = dataset[idx]

            # Prepare batch (add batch dimension)
            x = sample["x_continuous"].unsqueeze(0).to(device)
            positions = sample["positions"].unsqueeze(0).to(device)
            bluezone = sample["bluezone_info"].unsqueeze(0).to(device)
            whitezone = sample["whitezone_info"].unsqueeze(0).to(device)
            map_idx = sample["map_idx"].unsqueeze(0).to(device)
            num_squads = sample["num_squads"]
            squad_alive_mask = sample["squad_alive_mask"]

            # Forward pass: raw scores (log hazard)
            embeddings = backbone(x, positions, bluezone, whitezone, map_idx)
            raw_scores = head(embeddings).squeeze(0).cpu()

            # Apply temperature scaling
            scaled_scores = raw_scores / temperature

            # Apply softmax only over valid (alive) squads to get win probabilities
            # Mask out dead/padded squads with -inf before softmax
            masked_scores = scaled_scores.clone()
            masked_scores[~squad_alive_mask] = float("-inf")
            win_probs = torch.softmax(masked_scores[:num_squads], dim=0)

            for squad_idx in range(num_squads):
                results.append({
                    "match_id": match_id,
                    "time_point": time_point,
                    "squad_idx": squad_idx,
                    "win_probability": win_probs[squad_idx].item(),
                })

    df = pd.DataFrame(results)

    if save_predictions:
        if output_path is None:
            base_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(folder_path, f"{base_name}_predictions.csv")
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

    return df


def run_inference_on_folder(
    folder_path: str,
    file_list: List[str],
    checkpoint_path: str,
    config_path: str = None,
    device: str = "cpu",
    save_results: bool = True,
    output_path: str = None,
) -> Dict:
    """
    Run inference on multiple CSV files and compute phase-wise accuracy.

    Each match consists of phase-sampled time points (10 phases, non-uniform
    density); phase membership is derived from the time point value.

    Args:
        folder_path: Path to folder containing CSV files.
        file_list: List of CSV filenames to process.
        checkpoint_path: Path to model checkpoint.
        config_path: Path to config.json. If None, inferred.
        device: Device to use.
        save_results: Whether to save results to CSV.
        output_path: Output path for results. If None, auto-generated.

    Returns:
        Dictionary with phase-wise accuracies.
    """
    # Load model
    backbone, head, scaler_params, config = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )

    if scaler_params is None:
        raise ValueError(
            "Checkpoint does not contain scaler_params. "
            "Please use a checkpoint saved with scaler_params."
        )

    # Create dataset
    dataset = PGCDataset(
        folder_path=folder_path,
        file_list=file_list,
        continuous_features=CONTINUOUS_FEATURES,
        scaler_params=scaler_params,
    )

    # Evaluate
    if output_path is None and save_results:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        output_path = os.path.join(checkpoint_dir, "inference_results.csv")

    results = evaluate_test_set_by_match(
        backbone=backbone,
        head=head,
        test_dataset=dataset,
        device=device,
        save_path=output_path if save_results else None,
    )

    print_results(results, title="Inference Results")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference on CSV files using trained model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., checkpoints/exp/best.pt)",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to single CSV file for inference",
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default=None,
        help="Path to folder containing CSV files",
    )
    parser.add_argument(
        "--file_list",
        type=str,
        nargs="+",
        default=None,
        help="List of CSV filenames in folder_path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0, etc.)",
    )
    args = parser.parse_args()

    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = args.device

    if args.csv_path is not None:
        # Single file inference
        df = run_inference_on_csv(
            csv_path=args.csv_path,
            checkpoint_path=args.checkpoint_path,
            device=device,
            output_path=args.output_path,
        )
        print(f"\nPredictions shape: {df.shape}")
        print(df.head(20))

    elif args.folder_path is not None and args.file_list is not None:
        # Multiple files inference with phase-wise accuracy
        results = run_inference_on_folder(
            folder_path=args.folder_path,
            file_list=args.file_list,
            checkpoint_path=args.checkpoint_path,
            device=device,
            output_path=args.output_path,
        )

    else:
        parser.print_help()
        print("\nError: Either --csv_path or (--folder_path and --file_list) required")


