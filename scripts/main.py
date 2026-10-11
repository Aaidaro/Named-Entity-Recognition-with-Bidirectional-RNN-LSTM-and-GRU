"""Main entry point for the CoNLL-2003 BiRNN/BiLSTM/BiGRU NER pipeline."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, MutableMapping

import numpy as np
import torch
import yaml
from torch import nn

# Allow `python scripts/main.py` from the project root without absolute imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import create_dataloaders, describe_dataset
from models.model import SequenceTagger
from scripts.evaluate import build_classification_report, evaluate_model
from scripts.train import train_one_epoch
from utils.visualization import plot_history


def load_config(path: str | Path) -> MutableMapping:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_first_batch_shapes(data_bundle) -> None:
    """Print token and label tensor shapes for first train and validation batches."""
    train_tokens, train_labels, _ = next(iter(data_bundle.train_loader))
    valid_tokens, valid_labels, _ = next(iter(data_bundle.valid_loader))
    print(f"First train batch token shape: {tuple(train_tokens.shape)}")
    print(f"First train batch label shape: {tuple(train_labels.shape)}")
    print(f"First validation batch token shape: {tuple(valid_tokens.shape)}")
    print(f"First validation batch label shape: {tuple(valid_labels.shape)}")


def resolve_model_types(requested: str, configured: Iterable[str]) -> list[str]:
    """Resolve CLI model selection into a list of model types."""
    if requested == "all":
        return [model_type.lower() for model_type in configured]
    return [requested.lower()]


def train_model(model_type: str, config: MutableMapping, data_bundle, device: torch.device) -> Dict[str, object]:
    """Train and evaluate one recurrent model variant."""
    model_cfg = config["model"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    model = SequenceTagger(
        vocab_size=data_bundle.vocab.size(),
        num_labels=len(data_bundle.label_names),
        embedding_size=int(model_cfg.get("embedding_size", 64)),
        hidden_size=int(model_cfg.get("hidden_size", 64)),
        padding_idx=data_bundle.vocab.pad_id,
        rnn_type=model_type,
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=data_bundle.label_pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-3)))

    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
    best_valid_acc = -1.0
    save_dir = PROJECT_ROOT / output_cfg.get("save_dir", "models/saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"{model_type}_ner.pt"

    for epoch in range(1, int(train_cfg.get("num_epochs", 10)) + 1):
        train_metrics = train_one_epoch(
            model,
            data_bundle.train_loader,
            optimizer,
            criterion,
            device,
            ignore_index=data_bundle.label_pad_id,
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        )
        valid_metrics = evaluate_model(
            model,
            data_bundle.valid_loader,
            criterion,
            device,
            ignore_index=data_bundle.label_pad_id,
        )

        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_acc"].append(float(train_metrics["accuracy"]))
        history["valid_loss"].append(float(valid_metrics["loss"]))
        history["valid_acc"].append(float(valid_metrics["accuracy"]))

        print(
            f"[{model_type.upper()}] Epoch {epoch:02d} | "
            f"train loss={train_metrics['loss']:.4f}, train acc={train_metrics['accuracy']:.4f} | "
            f"valid loss={valid_metrics['loss']:.4f}, valid acc={valid_metrics['accuracy']:.4f}"
        )

        if float(valid_metrics["accuracy"]) > best_valid_acc:
            best_valid_acc = float(valid_metrics["accuracy"])
            torch.save(
                {
                    "model_type": model_type,
                    "model_state_dict": model.state_dict(),
                    "vocab": data_bundle.vocab.token_to_idx,
                    "label_names": data_bundle.label_names,
                    "config": dict(config),
                },
                checkpoint_path,
            )

    final_metrics = evaluate_model(
        model,
        data_bundle.valid_loader,
        criterion,
        device,
        ignore_index=data_bundle.label_pad_id,
    )
    report = build_classification_report(
        final_metrics["y_true"],
        final_metrics["y_pred"],
        data_bundle.label_names,
    )

    plots_dir = PROJECT_ROOT / output_cfg.get("plots_dir", "outputs/plots")
    reports_dir = PROJECT_ROOT / output_cfg.get("reports_dir", "outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    plot_history(history, plots_dir / f"{model_type}_curves.png", title=f"{model_type.upper()} NER training curves")
    with open(reports_dir / f"{model_type}_classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    print(f"\nClassification report for {model_type.upper()} on validation data:\n{report}")
    print(f"Saved best checkpoint to: {checkpoint_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved plot to: {(plots_dir / f'{model_type}_curves.png').relative_to(PROJECT_ROOT)}")

    return {"history": history, "report": report, "best_valid_acc": best_valid_acc}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Train BiRNN/BiLSTM/BiGRU models for CoNLL-2003 NER.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config relative to project root.")
    parser.add_argument("--model-type", choices=["all", "rnn", "lstm", "gru"], default="all")
    parser.add_argument("--subset", action="store_true", help="Force quick subset mode.")
    parser.add_argument("--full-data", action="store_true", help="Disable subset mode and use the full dataset.")
    return parser.parse_args()


def main() -> None:
    """Run dataset analysis, batching checks, training, plotting, and reporting."""
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    if args.subset:
        config["data"]["use_subset"] = True
    if args.full_data:
        config["data"]["use_subset"] = False

    set_seed(int(config.get("seed", 42)))
    device_name = "cuda" if torch.cuda.is_available() and config["training"].get("use_cuda", True) else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    data_bundle = create_dataloaders(config)
    print("\n=== Dataset analysis ===")
    print(describe_dataset(data_bundle.raw_dataset))
    print(f"Vocabulary size: {data_bundle.vocab.size()}")
    print(f"Token with id=0: {data_bundle.vocab.id_to_token(0)}")
    quick_id = data_bundle.vocab.token_to_id("quick")
    quick_note = "" if data_bundle.vocab.id_to_token(quick_id) == "quick" else " (mapped to <unk>)"
    print(f"ID of token 'quick': {quick_id}{quick_note}")

    print("\n=== First batch shapes ===")
    print_first_batch_shapes(data_bundle)

    model_types = resolve_model_types(args.model_type, config["training"].get("model_types", ["rnn", "lstm", "gru"]))
    results = {}
    for model_type in model_types:
        print(f"\n=== Training {model_type.upper()} model ===")
        set_seed(int(config.get("seed", 42)))
        results[model_type] = train_model(model_type, config, data_bundle, device)

    print("\n=== Summary ===")
    for model_type, result in results.items():
        print(f"{model_type.upper()}: best validation token accuracy = {result['best_valid_acc']:.4f}")


if __name__ == "__main__":
    main()
