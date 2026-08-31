import unittest
from unittest.mock import patch

import torch

from jet_gnn import DynamicJetGNN, FixedJetGNN, build_batched_knn_edges


class BatchedKnnTests(unittest.TestCase):
    def test_edges_never_cross_graph_boundaries(self):
        x = torch.tensor([[0.0], [1.0], [10.0], [11.0], [12.0]])
        batch = torch.tensor([0, 0, 1, 1, 1])

        edge_index = build_batched_knn_edges(x, batch, k=2)

        source_graphs = batch[edge_index[0]]
        target_graphs = batch[edge_index[1]]
        self.assertTrue(torch.equal(source_graphs, target_graphs))
        self.assertEqual(edge_index.size(1), 2 + 6)

    def test_small_graphs_reduce_k_and_singletons_are_safe(self):
        x = torch.tensor([[0.0], [3.0], [4.0]])
        batch = torch.tensor([0, 1, 1])

        edge_index = build_batched_knn_edges(x, batch, k=8)

        self.assertEqual(edge_index.size(1), 2)
        self.assertFalse(torch.any(edge_index == 0))


class GraphSemanticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(9, 4)
        self.pos = torch.tensor(
            [
                [0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.1, 0.0],
                [0.0, 2.0], [0.2, 2.0], [0.4, 2.0], [2.0, 2.0], [2.2, 2.0],
            ]
        )
        self.batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])

    def test_initial_graph_depends_only_on_pos(self):
        model = FixedJetGNN(num_features=4, k=2).eval()
        _, first = model(
            self.x, self.pos, self.batch, return_edge_indices=True
        )
        changed_x = self.x.clone()
        changed_x[:, 2:] = 1000 * torch.randn_like(changed_x[:, 2:])
        _, second = model(
            changed_x, self.pos, self.batch, return_edge_indices=True
        )
        expected = build_batched_knn_edges(self.pos, self.batch, k=2)
        self.assertTrue(torch.equal(first["geometric"], expected))
        self.assertTrue(torch.equal(second["geometric"], expected))

    def test_fixed_model_uses_same_geometric_graph_twice(self):
        model = FixedJetGNN(num_features=4, k=2).eval()
        with patch(
            "jet_gnn.build_batched_knn_edges",
            wraps=build_batched_knn_edges,
        ) as builder:
            _, edges = model(
                self.x, self.pos, self.batch, return_edge_indices=True
            )
        self.assertEqual(builder.call_count, 1)
        self.assertTrue(torch.equal(edges["geometric"], edges["second"]))

    def test_dynamic_first_graph_is_geometric_and_second_is_latent(self):
        model = DynamicJetGNN(num_features=4, k=2).eval()
        with patch(
            "jet_gnn.build_batched_knn_edges",
            wraps=build_batched_knn_edges,
        ) as builder:
            _, edges = model(
                self.x, self.pos, self.batch, return_edge_indices=True
            )
        self.assertEqual(builder.call_count, 2)
        first_space = builder.call_args_list[0].args[0]
        second_space = builder.call_args_list[1].args[0]
        self.assertTrue(torch.equal(first_space, self.pos))
        self.assertEqual(second_space.shape, (self.x.size(0), 64))
        expected_latent = build_batched_knn_edges(
            second_space, self.batch, k=2
        )
        self.assertTrue(torch.equal(edges["second"], expected_latent))

    def test_fixed_and_dynamic_parameter_counts_match(self):
        torch.manual_seed(17)
        fixed = FixedJetGNN(num_features=4, k=8)
        torch.manual_seed(17)
        dynamic = DynamicJetGNN(num_features=4, k=8)
        fixed_count = sum(parameter.numel() for parameter in fixed.parameters())
        dynamic_count = sum(
            parameter.numel() for parameter in dynamic.parameters()
        )
        self.assertEqual(fixed_count, dynamic_count)
        for name, fixed_value in fixed.state_dict().items():
            self.assertTrue(torch.equal(fixed_value, dynamic.state_dict()[name]))


class JetGnnTests(unittest.TestCase):
    def test_model_returns_one_logit_per_graph_and_backpropagates(self):
        torch.manual_seed(7)
        x = torch.randn(9, 4, requires_grad=True)
        pos = torch.randn(9, 2)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
        model = DynamicJetGNN(num_features=4, k=3)

        logits = model(x, pos, batch)
        logits.sum().backward()

        self.assertEqual(logits.shape, (2,))
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
