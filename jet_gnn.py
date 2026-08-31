"""Reusable graph-network components for quark/gluon jet classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import EdgeConv, global_mean_pool


def build_batched_knn_edges(x: Tensor, batch: Tensor, k: int) -> Tensor:
    """Build directed k-NN edges independently inside every graph in a batch.

    The neighbour search uses native PyTorch, so the model does not require
    ``DynamicEdgeConv``, ``pyg-lib`` or ``torch-cluster``. Neighbour selection
    is discrete and uses detached features; gradients still flow through the
    subsequent EdgeConv operation.
    """
    if x.ndim != 2:
        raise ValueError(
            f"x must have shape [num_nodes, num_features], got {x.shape}"
        )
    if batch.ndim != 1 or batch.numel() != x.size(0):
        raise ValueError("batch must contain one graph index per node")
    if k < 1:
        raise ValueError("k must be at least 1")

    edge_parts: list[Tensor] = []
    with torch.no_grad():
        for graph_id in torch.unique(batch, sorted=True):
            node_ids = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
            num_nodes = node_ids.numel()
            if num_nodes < 2:
                continue

            local_k = min(k, num_nodes - 1)
            features = x.detach().index_select(0, node_ids)
            distances = torch.cdist(features, features)
            distances.fill_diagonal_(float("inf"))
            neighbours = distances.topk(local_k, largest=False).indices

            target_local = torch.arange(
                num_nodes, device=x.device
            ).repeat_interleave(local_k)
            source_local = neighbours.reshape(-1)
            edge_parts.append(
                torch.stack(
                    (
                        node_ids.index_select(0, source_local),
                        node_ids.index_select(0, target_local),
                    ),
                    dim=0,
                )
            )

    if not edge_parts:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(edge_parts, dim=1)


class EdgeConvJetClassifier(nn.Module):
    """Shared two-layer EdgeConv architecture for controlled comparisons."""

    def __init__(self, num_features: int, k: int = 8, dynamic: bool = True) -> None:
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be at least 1")
        if k < 1:
            raise ValueError("k must be at least 1")

        self.k = k
        self.dynamic = dynamic
        self.conv1 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * num_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            ),
            aggr="max",
        )
        self.conv2 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * 64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            ),
            aggr="max",
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        *,
        return_edge_indices: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Classify jets while keeping geometry separate from node features."""
        if pos.ndim != 2 or pos.size(0) != x.size(0) or pos.size(1) != 2:
            raise ValueError("pos must have shape [num_nodes, 2] for dEta/dPhi")

        geometric_edges = build_batched_knn_edges(pos, batch, self.k)
        hidden = torch.relu(self.conv1(x, geometric_edges))
        second_edges = (
            build_batched_knn_edges(hidden, batch, self.k)
            if self.dynamic
            else geometric_edges
        )
        hidden = torch.relu(self.conv2(hidden, second_edges))

        graph_features = global_mean_pool(hidden, batch)
        logits = self.classifier(graph_features).squeeze(-1)
        if return_edge_indices:
            return logits, {
                "geometric": geometric_edges,
                "second": second_edges,
            }
        return logits


class FixedJetGNN(EdgeConvJetClassifier):
    """EdgeConv model that reuses the geometric graph in both layers."""

    def __init__(self, num_features: int, k: int = 8) -> None:
        super().__init__(num_features=num_features, k=k, dynamic=False)


class DynamicJetGNN(EdgeConvJetClassifier):
    """EdgeConv model that rebuilds the second graph in latent space."""

    def __init__(self, num_features: int, k: int = 8) -> None:
        super().__init__(num_features=num_features, k=k, dynamic=True)


# Preserve the established public name while giving it the corrected semantics.
JetGNN = DynamicJetGNN
