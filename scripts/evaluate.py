"""Evaluation utilities for token-level NER models."""

from __future__ import annotations

from typing import Dict, List

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import flatten_valid_predictions, token_accuracy


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ignore_index: int = -100,
) -> Dict[str, object]:
    """Evaluate a model and return loss, accuracy, labels, and predictions."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    all_true: List[int] = []
    all_pred: List[int] = []

    for input_ids, labels, lengths in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        mask = labels.ne(ignore_index)
        predictions = logits.argmax(dim=-1)
        total_correct += predictions.eq(labels).masked_select(mask).sum().item()
        total_tokens += mask.sum().item()
        total_loss += loss.item() * max(mask.sum().item(), 1)

        valid_labels, valid_predictions = flatten_valid_predictions(logits, labels, ignore_index)
        all_true.extend(valid_labels.tolist())
        all_pred.extend(valid_predictions.tolist())

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc = total_correct / max(total_tokens, 1)
    return {"loss": avg_loss, "accuracy": avg_acc, "y_true": all_true, "y_pred": all_pred}


def build_classification_report(
    y_true: List[int],
    y_pred: List[int],
    label_names: List[str],
) -> str:
    """Create a sklearn classification report with one row per NER label."""
    labels = list(range(len(label_names)))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        zero_division=0,
    )
