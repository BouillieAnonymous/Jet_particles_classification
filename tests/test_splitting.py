import unittest

import torch
from torch_geometric.data import Batch, Data

from src.splitting import (
    apply_standardization,
    event_ids,
    fit_standardization,
    split_by_event,
)


def graph(event_id, values):
    return Data(
        x=torch.tensor(values, dtype=torch.float32).reshape(-1, 1),
        pos=torch.tensor(
            [[float(index), 0.0] for index in range(len(values))]
        ),
        y=torch.tensor([event_id % 2]),
        run=torch.tensor([1]),
        lumi=torch.tensor([10]),
        event=torch.tensor([event_id]),
        event_id=torch.tensor([event_id]),
    )


class EventSplitTests(unittest.TestCase):
    def test_all_jets_from_an_event_stay_together(self):
        graphs = [graph(event, [event]) for event in range(20)]
        graphs.extend([graph(3, [30]), graph(7, [70])])
        split = split_by_event(graphs, seed=9)
        self.assertTrue(event_ids(split.train).isdisjoint(event_ids(split.validation)))
        self.assertTrue(event_ids(split.train).isdisjoint(event_ids(split.test)))
        self.assertTrue(event_ids(split.validation).isdisjoint(event_ids(split.test)))
        self.assertEqual(sum(map(len, (split.train, split.validation, split.test))), 22)

    def test_split_is_reproducible(self):
        graphs = [graph(event, [event]) for event in range(20)]
        first = split_by_event(graphs, seed=4)
        second = split_by_event(graphs, seed=4)
        self.assertEqual(event_ids(first.train), event_ids(second.train))

    def test_composite_identity_distinguishes_reused_event_numbers(self):
        graphs = [graph(event, [event]) for event in range(20)]
        reused = graph(3, [300])
        reused.run = torch.tensor([2])
        reused.lumi = torch.tensor([20])
        graphs.append(reused)
        split = split_by_event(graphs, seed=4)
        all_ids = event_ids(split.train) | event_ids(split.validation) | event_ids(split.test)
        self.assertIn((1, 10, 3), all_ids)
        self.assertIn((2, 20, 3), all_ids)
        self.assertEqual(len(all_ids), 21)


class StandardizationTests(unittest.TestCase):
    def test_statistics_are_fit_only_on_training_particles(self):
        train = [graph(1, [1.0, 3.0])]
        validation = [graph(2, [1000.0])]
        stats = fit_standardization(train)
        original_positions = [item.pos.clone() for item in train + validation]
        apply_standardization(train + validation, stats)
        self.assertAlmostEqual(stats.mean.item(), 2.0)
        self.assertAlmostEqual(train[0].x.mean().item(), 0.0)
        self.assertGreater(validation[0].x.item(), 100.0)
        for item, original_pos in zip(train + validation, original_positions):
            self.assertTrue(torch.equal(item.pos, original_pos))

    def test_batching_preserves_x_and_pos_semantics(self):
        graphs = [graph(1, [1.0, 2.0]), graph(2, [3.0, 4.0, 5.0])]
        batch = Batch.from_data_list(graphs)
        self.assertEqual(batch.x.shape, (5, 1))
        self.assertEqual(batch.pos.shape, (5, 2))
        self.assertTrue(torch.equal(batch.pos, torch.cat([g.pos for g in graphs])))


if __name__ == "__main__":
    unittest.main()
