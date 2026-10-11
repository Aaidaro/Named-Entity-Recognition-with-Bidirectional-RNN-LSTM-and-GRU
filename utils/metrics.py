"""Metrics for token-level sequence labeling."""

from __future__ import annotations

from typing import Tuple

import torch


def token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """Compute token accuracy while excluding padded label positions."""
    predictions = logits.argmax(dim=-1)
    mask = labels.ne(ignore_index)
    total = mask.sum().item()
    if total == 0:
        return 0.0
    correct = predictions.eq(labels).masked_select(mask).sum().item()
    return correct / total


def flatten_valid_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten predictions and labels after removing padded positions."""
    predictions = logits.argmax(dim=-1)
    mask = labels.ne(ignore_index)
    return labels[mask].detach().cpu(), predictions[mask].detach().cpu()
