"""
Training utilities for PGC models with DDP support.
"""

import os
import json
from datetime import datetime
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data import CONTINUOUS_FEATURES, PGCDataset
from src.data.dataset import collate_fn
from src.data.dataset_v2 import (
    PGCDatasetV2,
    collate_fn as collate_fn_v2,
    COMPUTED_FEATURES,
)
from src.models import TransformerBackbone
from src.models.heads import get_head
from src.training.losses import get_loss_fn


def set_seed(seed: int):
    """Seed all random number generators for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_ddp(rank: int, world_size: int):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP process group."""
    dist.destroy_process_group()


class Trainer:
    """
    Trainer for PGC survival time prediction with DDP support.

    Args:
        backbone: TransformerBackbone model.
        head: SurvivalTimeHead model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        device: Device to use for training.
        checkpoint_dir: Directory to save checkpoints.
        rank: Process rank for DDP. None for single GPU.
        world_size: Total number of processes for DDP.
    """

    def __init__(
        self,
        backbone: TransformerBackbone,
        head: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_type: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        rank: int = None,
        world_size: int = 1,
        scaler_params: Dict = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_ddp = rank is not None
        self.is_main_process = (rank is None) or (rank == 0)

        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.loss_type = loss_type.lower()

        # Move models to device
        self.backbone = backbone.to(device)
        self.head = head.to(device)

        # Wrap with DDP if needed
        if self.is_ddp:
            self.backbone = DDP(self.backbone, device_ids=[rank])
            self.head = DDP(self.head, device_ids=[rank])

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function based on loss_type
        self.criterion = get_loss_fn(loss_type)

        # Optimizer
        params = (
            list(self.backbone.parameters()) +
            list(self.head.parameters())
        )
        self.optimizer = optim.AdamW(
            params, lr=lr, weight_decay=weight_decay
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "best_val_loss": float("inf"),
            "best_epoch": 0,
        }

        # Scaler parameters for inference
        self.scaler_params = scaler_params

        # Create checkpoint directory (only main process)
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def _get_backbone(self):
        """Get backbone model (unwrap DDP if needed)."""
        if self.is_ddp:
            return self.backbone.module
        return self.backbone

    def _get_head(self):
        """Get head model (unwrap DDP if needed)."""
        if self.is_ddp:
            return self.head.module
        return self.head

    def _forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through backbone and head."""
        x = batch["x_continuous"].to(self.device)
        positions = batch["positions"].to(self.device)
        bluezone = batch["bluezone_info"].to(self.device)
        whitezone = batch["whitezone_info"].to(self.device)
        map_idx = batch["map_idx"].to(self.device)

        embeddings = self.backbone(x, positions, bluezone, whitezone, map_idx)
        predictions = self.head(embeddings)

        return predictions

    def _compute_masked_loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        alive_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for valid (alive and non-padded) squads.

        Supports MSE, Cox, and Cross-entropy losses.

        Args:
            pred: Predictions (batch, num_squads)
            y: Targets (batch, num_squads)
            alive_mask: (batch, num_squads), True if squad is alive
            padding_mask: (batch, num_squads), True if valid (non-padded)

        Returns:
            Scalar loss value.
        """
        # Combine masks: valid = alive AND non-padded
        valid_mask = alive_mask & padding_mask

        # Use the loss function which handles masking internally
        return self.criterion(pred, y, valid_mask)

    def train_epoch(self, epoch: int, num_epochs: int = None) -> float:
        """Train for one epoch."""
        from tqdm import tqdm

        self.backbone.train()
        self.head.train()

        # Set epoch for distributed sampler
        if self.is_ddp and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0

        # Progress bar with ETA (only on main process)
        desc = f"Epoch {epoch}"
        if num_epochs is not None:
            desc = f"Epoch {epoch}/{num_epochs}"

        loader = self.train_loader
        if self.is_main_process:
            loader = tqdm(
                self.train_loader,
                desc=desc,
                leave=False,
                dynamic_ncols=True,
            )

        for batch in loader:
            y = batch["y"].to(self.device)
            alive_mask = batch["squad_alive_mask"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)

            self.optimizer.zero_grad()
            pred = self._forward(batch)
            loss = self._compute_masked_loss(pred, y, alive_mask, padding_mask)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with current loss
            if self.is_main_process and hasattr(loader, 'set_postfix'):
                loader.set_postfix(loss=f"{total_loss/num_batches:.4f}")

        avg_loss = total_loss / num_batches

        # Reduce loss across all processes
        if self.is_ddp:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        self.backbone.eval()
        self.head.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            y = batch["y"].to(self.device)
            alive_mask = batch["squad_alive_mask"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            pred = self._forward(batch)
            loss = self._compute_masked_loss(pred, y, alive_mask, padding_mask)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Reduce loss across all processes
        if self.is_ddp:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return

        checkpoint = {
            "epoch": epoch,
            "backbone_state_dict": self._get_backbone().state_dict(),
            "head_state_dict": self._get_head().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "scaler_params": self.scaler_params,
        }

        # Save latest checkpoint
        path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self._get_backbone().load_state_dict(checkpoint["backbone_state_dict"])
        self._get_head().load_state_dict(checkpoint["head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]

        return checkpoint["epoch"]

    def train(
        self,
        num_epochs: int,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            num_epochs: Number of epochs to train.
            patience: Early stopping patience. Training stops if val_loss
                      doesn't improve for this many epochs.
            verbose: Whether to print progress.

        Returns:
            Training history dictionary.
        """
        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch, num_epochs)
            val_loss = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Check for best model
            is_best = val_loss < self.history["best_val_loss"]
            if is_best:
                self.history["best_val_loss"] = val_loss
                self.history["best_epoch"] = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            if verbose and self.is_main_process:
                best_marker = " *" if is_best else ""
                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}{best_marker}"
                )

            # Early stopping check
            if epochs_without_improvement >= patience:
                if verbose and self.is_main_process:
                    print(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(no improvement for {patience} epochs)"
                    )
                break

        # Record early stopping info
        self.history["stopped_epoch"] = epoch
        self.history["early_stopped"] = epochs_without_improvement >= patience

        # Save final history (only main process)
        if self.is_main_process:
            history_path = os.path.join(self.checkpoint_dir, "history.json")
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)

        return self.history


def get_ddp_dataloaders(
    folder_path: str,
    split_csv_path: str,
    continuous_features=CONTINUOUS_FEATURES,
    batch_size: int = 32,
    num_workers: int = 4,
    rank: int = None,
    world_size: int = 1,
    toy_ratio: float = 1.0,
    use_dataset_v2: bool = True,
    seed: int = 42,
):
    """
    Create DataLoaders with DistributedSampler for DDP.

    Args:
        folder_path: Path to data folder.
        split_csv_path: Path to split CSV file.
        continuous_features: List of continuous features.
        batch_size: Batch size per GPU.
        num_workers: Number of data loading workers.
        rank: Process rank for DDP.
        world_size: Total number of processes.
        toy_ratio: Ratio of data to use (0.0-1.0). Default 1.0 uses all data.
        use_dataset_v2: If True, use PGCDatasetV2 with distance features.
        seed: Random seed for data subsampling.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler_params).
    """
    import pandas as pd

    # Select dataset class and collate function based on version
    if use_dataset_v2:
        DatasetClass = PGCDatasetV2
        collate_function = collate_fn_v2
        print("Using PGCDatasetV2 (with distance features)")
    else:
        DatasetClass = PGCDataset
        collate_function = collate_fn
        print("Using PGCDataset (original)")

    # Load split information
    split_df = pd.read_csv(split_csv_path)
    train_files = split_df[split_df["split"] == "train"]["filename"].tolist()
    val_files = split_df[split_df["split"] == "val"]["filename"].tolist()
    test_files = split_df[split_df["split"] == "test"]["filename"].tolist()
    print(f"toy_ratio: {toy_ratio}")

    # Apply toy_ratio to use subset of data
    if toy_ratio < 1.0:
        import random
        random.seed(seed)
        n_train = max(1, int(len(train_files) * toy_ratio))
        n_val = max(1, int(len(val_files) * toy_ratio))
        n_test = max(1, int(len(test_files) * toy_ratio))
        train_files = random.sample(train_files, n_train)
        val_files = random.sample(val_files, n_val)
        test_files = random.sample(test_files, n_test)
        print(f"Toy mode: using {toy_ratio*100:.0f}% of data")
        print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Create datasets using selected class
    train_dataset = DatasetClass(
        folder_path=folder_path,
        file_list=train_files,
        continuous_features=continuous_features,
        scaler_params=None,
    )

    # Compute scaler from train
    scaler_params = train_dataset.compute_scaler_params()
    train_dataset.scaler_params = scaler_params
    train_dataset._apply_scaling()

    val_dataset = DatasetClass(
        folder_path=folder_path,
        file_list=val_files,
        continuous_features=continuous_features,
        scaler_params=scaler_params,
    )

    test_dataset = DatasetClass(
        folder_path=folder_path,
        file_list=test_files,
        continuous_features=continuous_features,
        scaler_params=scaler_params,
    )

    # Create samplers
    is_ddp = rank is not None

    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    # Create data loaders with custom collate_fn for variable squad sizes
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_function,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_function,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_function,
    )

    return train_loader, val_loader, test_loader, scaler_params


def run_experiment(
    folder_path: str,
    split_csv_path: str,
    checkpoint_dir: str = None,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 100,
    patience: int = 10,
    num_workers: int = 4,
    device: str = "cpu",
    rank: int = None,
    world_size: int = 1,
    toy_ratio: float = 1.0,
    loss_type: str = "mse",
    use_dataset_v2: bool = True,
    seed: int = 42,
    encoder_type: str = "transformer",
) -> Dict:
    """
    Run a complete training experiment (single GPU or DDP).

    Args:
        folder_path: Path to data folder.
        split_csv_path: Path to split CSV file.
        checkpoint_dir: Directory for checkpoints.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout probability.
        batch_size: Batch size per GPU.
        lr: Learning rate.
        weight_decay: Weight decay.
        num_epochs: Number of training epochs.
        patience: Early stopping patience.
        num_workers: Number of data loading workers.
        device: Device to use (ignored for DDP, uses rank).
        rank: Process rank for DDP. None for single GPU.
        toy_ratio: Ratio of data to use (0.0-1.0). Default 1.0 uses all data.
        world_size: Total number of processes for DDP.
        loss_type: Loss function type ('mse', 'cox', 'ce').
        use_dataset_v2: If True, use PGCDatasetV2 with distance features.
        seed: Random seed for reproducibility.
        encoder_type: 'transformer' or 'mlp' (Table 2 encoder ablation).

    Returns:
        Training history dictionary.
    """
    set_seed(seed)

    is_ddp = rank is not None
    is_main_process = (rank is None) or (rank == 0)

    # Set device
    if is_ddp:
        device = f"cuda:{rank}"
    elif device == "cuda":
        device = "cuda:0"

    # Auto-generate checkpoint directory with model configuration
    if checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_str = (
            f"emb{embed_dim}_head{num_heads}_layer{num_layers}_"
            f"drop{dropout}_lr{lr}_bs{batch_size}"
        )
        checkpoint_dir = f"checkpoints/{timestamp}_{config_str}"

    # Create data loaders
    train_loader, val_loader, test_loader, scaler_params = get_ddp_dataloaders(
        folder_path=folder_path,
        split_csv_path=split_csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
        toy_ratio=toy_ratio,
        use_dataset_v2=use_dataset_v2,
        seed=seed,
    )

    # Create models
    # input_dim includes computed features if using dataset_v2
    if use_dataset_v2:
        input_dim = len(CONTINUOUS_FEATURES) + len(COMPUTED_FEATURES)
    else:
        input_dim = len(CONTINUOUS_FEATURES)

    backbone = TransformerBackbone(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        encoder_type=encoder_type,
    )

    # Get appropriate head based on loss type
    head = get_head(loss_type, embed_dim=embed_dim, dropout=dropout)

    # Create trainer
    trainer = Trainer(
        backbone=backbone,
        head=head,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_type=loss_type,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        checkpoint_dir=checkpoint_dir,
        rank=rank,
        world_size=world_size,
        scaler_params=scaler_params,
    )

    # Save experiment config (only main process)
    if is_main_process:
        config = {
            "folder_path": folder_path,
            "split_csv_path": split_csv_path,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "patience": patience,
            "num_workers": num_workers,
            "world_size": world_size,
            "input_dim": input_dim,
            "loss_type": loss_type,
            "use_dataset_v2": use_dataset_v2,
            "seed": seed,
            "encoder_type": encoder_type,
        }

        os.makedirs(checkpoint_dir, exist_ok=True)
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        backbone_params = sum(p.numel() for p in backbone.parameters())
        head_params = sum(p.numel() for p in head.parameters())
        print(f"Experiment directory: {checkpoint_dir}")
        print(f"World size: {world_size}")
        print(f"Backbone params: {backbone_params:,}")
        print(f"Head params: {head_params:,}")
        print("-" * 50)

    # Synchronize before training
    if is_ddp:
        dist.barrier()

    # Train
    history = trainer.train(num_epochs=num_epochs, patience=patience)

    if is_main_process:
        print("-" * 50)
        print(
            f"Best Val Loss: {history['best_val_loss']:.4f} "
            f"(Epoch {history['best_epoch']})"
        )

        # Load best model for evaluation
        best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            trainer._get_backbone().load_state_dict(
                checkpoint["backbone_state_dict"]
            )
            trainer._get_head().load_state_dict(
                checkpoint["head_state_dict"]
            )

            # Evaluate on test set
            from src.training.evaluation import (
                evaluate_test_set_by_match,
                print_results,
            )

            # Get test dataset
            test_dataset = test_loader.dataset

            results_path = os.path.join(checkpoint_dir, "test_results.csv")
            results = evaluate_test_set_by_match(
                backbone=trainer._get_backbone(),
                head=trainer._get_head(),
                test_dataset=test_dataset,
                device=device,
                save_path=results_path,
            )

            print_results(results, title="Test Set Evaluation")
            history["test_results"] = results

    return history


def run_ddp_worker(
    rank: int,
    world_size: int,
    folder_path: str,
    split_csv_path: str,
    checkpoint_dir: str,
    kwargs_dict: dict,
):
    """Worker function for DDP training."""
    setup_ddp(rank, world_size)

    run_experiment(
        folder_path=folder_path,
        split_csv_path=split_csv_path,
        checkpoint_dir=checkpoint_dir,
        rank=rank,
        world_size=world_size,
        **kwargs_dict,
    )

    cleanup_ddp()


def run_ddp_experiment(
    folder_path: str,
    split_csv_path: str,
    checkpoint_dir: str = None,
    world_size: int = 8,
    **kwargs,
):
    """
    Launch DDP training across multiple GPUs.

    Args:
        folder_path: Path to data folder.
        split_csv_path: Path to split CSV file.
        checkpoint_dir: Directory for checkpoints.
        world_size: Number of GPUs to use.
        **kwargs: Additional arguments for run_experiment.
    """
    import torch.multiprocessing as mp

    # Auto-generate checkpoint directory with model configuration
    if checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        embed_dim = kwargs.get('embed_dim', 128)
        num_heads = kwargs.get('num_heads', 4)
        num_layers = kwargs.get('num_layers', 2)
        dropout = kwargs.get('dropout', 0.1)
        lr = kwargs.get('lr', 1e-3)
        batch_size = kwargs.get('batch_size', 32)
        config_str = (
            f"emb{embed_dim}_head{num_heads}_layer{num_layers}_"
            f"drop{dropout}_lr{lr}_bs{batch_size}"
        )
        checkpoint_dir = f"checkpoints/{timestamp}_{config_str}"

    mp.spawn(
        run_ddp_worker,
        args=(
            world_size,
            folder_path,
            split_csv_path,
            checkpoint_dir,
            kwargs,  # Pass kwargs to worker
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="data")
    parser.add_argument("--split_csv_path", type=str, default="data/split.csv")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--toy_ratio", type=float, default=1.0,
                        help="Ratio of data to use (0.0-1.0). Default 1.0 uses all data.")
    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "cox", "ce", "rank_cox", "weighted_cox", 
                                 "concordance", "survival_ce"],
                        help="Loss type: 'mse', 'cox', 'ce', 'rank_cox', 'weighted_cox', "
                             "'concordance', 'survival_ce'")
    parser.add_argument("--use_dataset_v2", type=str, default="true",
                        choices=["true", "false"],
                        help="Use PGCDatasetV2 with distance features (default: true)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--encoder", type=str, default="transformer",
                        choices=["transformer", "mlp"],
                        help="Encoder type: 'transformer' (self-attention) or "
                             "'mlp' (no cross-squad attention; Table 2 ablation)")
    args = parser.parse_args()
    
    # Convert string to boolean
    args.use_dataset_v2 = args.use_dataset_v2.lower() == "true"

    if args.world_size > 1:
        # Multi-GPU DDP training
        run_ddp_experiment(
            folder_path=args.folder_path,
            split_csv_path=args.split_csv_path,
            checkpoint_dir=args.checkpoint_dir,
            world_size=args.world_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            toy_ratio=args.toy_ratio,
            loss_type=args.loss_type,
            use_dataset_v2=args.use_dataset_v2,
            seed=args.seed,
            encoder_type=args.encoder,
        )
    else:
        # Single GPU training
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_experiment(
            folder_path=args.folder_path,
            split_csv_path=args.split_csv_path,
            checkpoint_dir=args.checkpoint_dir,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            device=device,
            toy_ratio=args.toy_ratio,
            loss_type=args.loss_type,
            use_dataset_v2=args.use_dataset_v2,
            seed=args.seed,
            encoder_type=args.encoder,
        )

