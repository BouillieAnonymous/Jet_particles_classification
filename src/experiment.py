"""Reproducible runner for controlled fixed/dynamic EdgeConv experiments."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from jet_gnn import DynamicJetGNN, FixedJetGNN
from src.dataset import find_root_path, load_jet_graphs
from src.features import feature_names
from src.metrics import binary_metrics
from src.splitting import (
    apply_standardization,
    fit_standardization,
    split_by_event,
)

METRICS_COLUMNS = (
    "experiment_id",
    "timestamp",
    "model",
    "features",
    "seed",
    "split_seed",
    "k",
    "max_epochs",
    "patience",
    "train_size",
    "validation_size",
    "test_size",
    "best_epoch",
    "epochs_trained",
    "stopped_early",
    "validation_auc",
    "test_auc",
    "test_balanced_accuracy",
    "training_time",
    "parameter_count",
)


@dataclass(frozen=True)
class ExperimentConfig:
    repository_root: Path
    root_path: Path | None = None
    results_dir: Path = Path("results")
    model: str = "dynamic"
    features: str = "full"
    seed: int = 1
    split_seed: int = 42
    k: int = 8
    max_epochs: int = 50
    patience: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_entries: int | None = None
    device: str = "auto"

    def validate(self) -> None:
        if self.model not in {"fixed", "dynamic"}:
            raise ValueError("model must be either 'fixed' or 'dynamic'")
        feature_names(self.features)
        if self.k < 1 or self.max_epochs < 1 or self.batch_size < 1:
            raise ValueError("k, max epochs and batch size must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _loader(graphs, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    logits_parts = []
    label_parts = []
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(batch.x, batch.pos, batch.batch)
            labels = batch.y.float()
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        logits_parts.append(logits.detach().cpu())
        label_parts.append(batch.y.detach().cpu())
    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts).numpy()
    probabilities = torch.sigmoid(logits).numpy()
    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = total_loss / len(labels)
    return metrics


def _jsonable_config(config: ExperimentConfig, root_path: Path) -> dict:
    payload = asdict(config)
    payload["feature_names"] = list(feature_names(config.features))
    payload["repository_root"] = str(config.repository_root.resolve())
    payload["root_path"] = str(root_path)
    results_dir = config.results_dir
    if not results_dir.is_absolute():
        results_dir = config.repository_root / results_dir
    payload["results_dir"] = str(results_dir.resolve())
    return payload


def append_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=METRICS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: metrics[column] for column in METRICS_COLUMNS})


def build_model(model_name: str, num_features: int, k: int):
    """Construct either controlled architecture from the same dimensions."""
    model_types = {"fixed": FixedJetGNN, "dynamic": DynamicJetGNN}
    try:
        return model_types[model_name](num_features=num_features, k=k)
    except KeyError as error:
        raise ValueError(f"Unknown model {model_name!r}") from error


def run_experiment(config: ExperimentConfig) -> dict:
    config.validate()
    set_global_seed(config.seed)
    root_path = find_root_path(config.repository_root, config.root_path)
    graphs = load_jet_graphs(root_path, config.features, config.max_entries)
    split = split_by_event(graphs, seed=config.split_seed)
    standardization = fit_standardization(split.train)
    for subset in (split.train, split.validation, split.test):
        apply_standardization(subset, standardization)

    train_loader = _loader(split.train, config.batch_size, True, config.seed)
    validation_loader = _loader(
        split.validation, config.batch_size, False, config.seed
    )
    test_loader = _loader(split.test, config.batch_size, False, config.seed)
    device = _device(config.device)
    model = build_model(
        config.model, len(feature_names(config.features)), config.k
    ).to(device)

    train_targets = torch.cat([graph.y for graph in split.train])
    num_positive = int((train_targets == 1).sum())
    num_negative = int((train_targets == 0).sum())
    pos_weight = num_negative / max(num_positive, 1)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        train_metrics = _epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_metrics = _epoch(
            model, validation_loader, loss_function, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_auc": train_metrics["auc"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_auc": validation_metrics["auc"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
            }
        )
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} "
            f"AUC {train_metrics['auc']:.4f} | validation loss "
            f"{validation_metrics['loss']:.4f} AUC "
            f"{validation_metrics['auc']:.4f}"
        )
        if epochs_without_improvement >= config.patience:
            stopped_early = True
            print(
                f"Early stopping after {epoch} epochs; "
                f"best validation loss was at epoch {best_epoch}."
            )
            break

    training_time = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Training completed without a best model state")
    model.load_state_dict(best_state)
    validation_metrics = _epoch(model, validation_loader, loss_function, device)
    test_metrics = _epoch(model, test_loader, loss_function, device)

    timestamp = datetime.now(timezone.utc)
    experiment_id = (
        f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}_"
        f"{config.model}_{config.features}_seed{config.seed}"
    )
    results_dir = config.results_dir
    if not results_dir.is_absolute():
        results_dir = config.repository_root / results_dir
    run_dir = results_dir / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metrics = {
        "experiment_id": experiment_id,
        "timestamp": timestamp.isoformat(),
        "model": config.model,
        "features": config.features,
        "seed": config.seed,
        "split_seed": config.split_seed,
        "k": config.k,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "train_size": len(split.train),
        "validation_size": len(split.validation),
        "test_size": len(split.test),
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "stopped_early": stopped_early,
        "validation_auc": validation_metrics["auc"],
        "test_auc": test_metrics["auc"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "training_time": training_time,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    (run_dir / "config.json").write_text(
        json.dumps(_jsonable_config(config, root_path), indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with (run_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    torch.save(
        {
            "model_state_dict": best_state,
            "feature_mean": standardization.mean,
            "feature_std": standardization.std,
            "config": _jsonable_config(config, root_path),
        },
        run_dir / "model.pt",
    )
    append_metrics(results_dir / "metrics.csv", metrics)
    return metrics
