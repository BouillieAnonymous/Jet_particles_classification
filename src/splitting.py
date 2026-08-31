"""Event-safe dataset splitting and train-only feature standardisation."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data

EventIdentity = tuple[int, int, int]


@dataclass(frozen=True)
class EventSplit:
    train: list[Data]
    validation: list[Data]
    test: list[Data]


@dataclass(frozen=True)
class Standardization:
    mean: Tensor
    std: Tensor


def split_by_event(
    graphs: list[Data],
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> EventSplit:
    """Assign all jets from an event to exactly one dataset subset."""
    if not graphs:
        raise ValueError("Cannot split an empty graph list")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must lie strictly between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one")

    identities = sorted({event_identity(graph) for graph in graphs})
    random.Random(seed).shuffle(identities)
    num_train = int(len(identities) * train_fraction)
    num_validation = int(len(identities) * validation_fraction)
    if num_train == 0 or num_validation == 0:
        raise ValueError("Not enough unique events for non-empty train/validation sets")

    train_ids = set(identities[:num_train])
    validation_ids = set(identities[num_train : num_train + num_validation])
    test_ids = set(identities[num_train + num_validation :])
    if not test_ids:
        raise ValueError("Not enough unique events for a non-empty test set")

    split = EventSplit(
        train=[g for g in graphs if event_identity(g) in train_ids],
        validation=[
            g for g in graphs if event_identity(g) in validation_ids
        ],
        test=[g for g in graphs if event_identity(g) in test_ids],
    )
    assert_event_disjoint(split)
    if sum(map(len, (split.train, split.validation, split.test))) != len(graphs):
        raise AssertionError("Some graphs were lost during event splitting")
    return split


def event_identity(graph: Data) -> EventIdentity:
    """Return the explicit CMS composite event identity."""
    missing = [name for name in ("run", "lumi", "event") if not hasattr(graph, name)]
    if missing:
        raise ValueError(f"Graph is missing composite event metadata: {missing}")
    return (
        int(graph.run.item()),
        int(graph.lumi.item()),
        int(graph.event.item()),
    )


def event_ids(graphs: list[Data]) -> set[EventIdentity]:
    return {event_identity(graph) for graph in graphs}


def assert_event_disjoint(split: EventSplit) -> None:
    train_ids = event_ids(split.train)
    validation_ids = event_ids(split.validation)
    test_ids = event_ids(split.test)
    if not train_ids.isdisjoint(validation_ids):
        raise AssertionError("Event leakage between train and validation sets")
    if not train_ids.isdisjoint(test_ids):
        raise AssertionError("Event leakage between train and test sets")
    if not validation_ids.isdisjoint(test_ids):
        raise AssertionError("Event leakage between validation and test sets")


def fit_standardization(train_graphs: list[Data]) -> Standardization:
    """Fit node-feature statistics using training particles only."""
    if not train_graphs:
        raise ValueError("Cannot fit standardization on an empty training set")
    particles = torch.cat([graph.x for graph in train_graphs], dim=0)
    return Standardization(
        mean=particles.mean(dim=0),
        std=particles.std(dim=0).clamp_min(1e-6),
    )


def apply_standardization(graphs: list[Data], stats: Standardization) -> None:
    for graph in graphs:
        graph.x = (graph.x - stats.mean) / stats.std
