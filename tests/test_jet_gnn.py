import unittest

import torch

from jet_gnn import JetGNN, build_batched_knn_edges


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


class JetGnnTests(unittest.TestCase):
    def test_model_returns_one_logit_per_graph_and_backpropagates(self):
        torch.manual_seed(7)
        x = torch.randn(9, 4, requires_grad=True)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
        model = JetGNN(num_features=4, k=3)

        logits = model(x, batch)
        logits.sum().backward()

        self.assertEqual(logits.shape, (2,))
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
