"""
Evaluation metrics for PGC win-probability models.

Metrics:
    - Accuracy: Whether the argmax winner matches the actual winner.
    - C-index (concordance index): Ranking quality over survival times.
    - IBS (Integrated Brier Score): Probabilistic error integrated over time.
    - ECE (Expected Calibration Error): Calibration of predicted probabilities.
    - Log Loss: Cross-entropy loss for winner probability.

The three primary metrics reported in the paper are Accuracy, C-index, and IBS.
ECE and Log Loss are included for calibration diagnostics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict


def compute_winner_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """
    Accuracy of winner prediction: argmax(pred) == argmax(target).

    Args:
        pred: (batch, num_squads) - raw scores.
        target: (batch, num_squads) - ground-truth scores (e.g., survival times).
        valid_mask: (batch, num_squads) - True for valid squads.

    Returns:
        Accuracy in [0, 1].
    """
    pred_masked = pred.clone()
    target_masked = target.clone()
    pred_masked[~valid_mask] = float('-inf')
    target_masked[~valid_mask] = float('-inf')

    pred_winners = pred_masked.argmax(dim=-1)
    target_winners = target_masked.argmax(dim=-1)

    return (pred_winners == target_winners).float().mean().item()


def compute_winner_probabilities(
    pred: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert raw scores to winner probabilities via masked softmax.
    Invalid positions are assigned zero probability.
    """
    pred_masked = pred.clone()
    pred_masked[~valid_mask] = float('-inf')
    probs = F.softmax(pred_masked, dim=-1)
    probs[~valid_mask] = 0.0
    return probs


def compute_log_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-7,
) -> float:
    """
    Cross-entropy loss against the true winner.
    """
    batch_size = pred.size(0)
    probs = compute_winner_probabilities(pred, valid_mask)

    target_masked = target.clone()
    target_masked[~valid_mask] = float('-inf')
    target_winners = target_masked.argmax(dim=-1)

    winner_probs = probs[torch.arange(batch_size), target_winners]
    winner_probs = torch.clamp(winner_probs, min=eps, max=1.0 - eps)
    return -torch.log(winner_probs).mean().item()


def compute_ece(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (Eq. 15).

    Bins predictions by maximum confidence and measures the gap between
    mean confidence and mean empirical accuracy within each bin.
    The paper uses B = 15 bins.
    """
    batch_size = pred.size(0)
    probs = compute_winner_probabilities(pred, valid_mask)

    target_masked = target.clone()
    target_masked[~valid_mask] = float('-inf')
    target_winners = target_masked.argmax(dim=-1)

    pred_winners = probs.argmax(dim=-1)
    confidences = probs.max(dim=-1).values
    correct = (pred_winners == target_winners).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = 0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        n = in_bin.sum().item()
        if n > 0:
            avg_conf = confidences[in_bin].mean().item()
            avg_acc = correct[in_bin].mean().item()
            ece += n * abs(avg_acc - avg_conf)
            total += n

    return ece / total if total > 0 else 0.0


def compute_c_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """
    Concordance index (C-index) for ranking survival times.

    For each pair of valid squads (i, j) with distinct target survival times,
    a pair is *concordant* if the predicted ordering matches the true ordering,
    i.e. pred[i] > pred[j] when target[i] > target[j]. Pairs with tied
    predictions contribute 0.5 (standard convention). Ties in the target are
    excluded from the denominator.

    Higher `pred` is expected to correspond to longer survival (later death).

    Args:
        pred: (batch, num_squads) - raw scores; higher = longer survival.
        target: (batch, num_squads) - ground-truth survival times.
        valid_mask: (batch, num_squads) - True for valid squads.

    Returns:
        C-index in [0, 1]; 0.5 is random, 1.0 is perfect.
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    valid_mask = valid_mask.detach().cpu().numpy().astype(bool)

    concordant = 0.0
    comparable = 0

    for b in range(pred.shape[0]):
        mask = valid_mask[b]
        p = pred[b][mask]
        t = target[b][mask]
        n = len(t)
        for i in range(n):
            for j in range(i + 1, n):
                if t[i] == t[j]:
                    continue
                comparable += 1
                if t[i] > t[j]:
                    if p[i] > p[j]:
                        concordant += 1
                    elif p[i] == p[j]:
                        concordant += 0.5
                else:  # t[j] > t[i]
                    if p[j] > p[i]:
                        concordant += 1
                    elif p[i] == p[j]:
                        concordant += 0.5

    return concordant / comparable if comparable > 0 else 0.0


def compute_integrated_brier_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """
    Integrated Brier Score (IBS) for winner probability predictions.

    For each time point, the Brier score is the squared error between the
    predicted winner probability and the one-hot winner indicator, summed
    over the squads in the risk set (Eq. 18). The IBS averages this
    per-time-point Brier score across all valid time points (batch dimension).

    Args:
        pred: (batch, num_squads) - raw scores; softmax produces probabilities.
        target: (batch, num_squads) - ground-truth survival times; argmax
            indicates the actual winner.
        valid_mask: (batch, num_squads) - True for valid squads.

    Returns:
        IBS (lower is better). Because squared errors are summed over the
        risk set, the value may exceed 1.
    """
    probs = compute_winner_probabilities(pred, valid_mask)

    target_masked = target.clone()
    target_masked[~valid_mask] = float('-inf')
    target_winners = target_masked.argmax(dim=-1)

    winner_onehot = torch.zeros_like(probs)
    winner_onehot[torch.arange(probs.size(0)), target_winners] = 1.0

    brier_per_squad = (probs - winner_onehot) ** 2
    brier_per_squad = brier_per_squad * valid_mask.float()

    # Sum over the risk set per time point, then average over time (batch).
    brier_per_sample = brier_per_squad.sum(dim=-1)
    return brier_per_sample.mean().item()


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics in a single call.

    Returns:
        Dict with keys: accuracy, c_index, ibs, ece, log_loss.
    """
    return {
        'accuracy': compute_winner_accuracy(pred, target, valid_mask),
        'c_index': compute_c_index(pred, target, valid_mask),
        'ibs': compute_integrated_brier_score(pred, target, valid_mask),
        'ece': compute_ece(pred, target, valid_mask),
        'log_loss': compute_log_loss(pred, target, valid_mask),
    }
