"""Training loop for sequence tagging models."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    ignore_index: int = -100,
    grad_clip: float | None = 1.0,
) -> Dict[str, float]:
    """Train ``model`` for one epoch and return average loss and token accuracy."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for input_ids, labels, lengths in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        mask = labels.ne(ignore_index)
        predictions = logits.argmax(dim=-1)
        total_correct += predictions.eq(labels).masked_select(mask).sum().item()
        token_count = mask.sum().item()
        total_tokens += token_count
        total_loss += loss.item() * max(token_count, 1)

    return {
        "loss": total_loss / max(total_tokens, 1),
        "accuracy": total_correct / max(total_tokens, 1),
    }
