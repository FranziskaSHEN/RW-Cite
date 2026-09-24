import unittest

import torch

from experiments.ewm_top10.common import arxiv_month, chronological_split, unique_top_k
from experiments.ewm_top10.metrics import metrics_at_k
from experiments.ewm_top10.modeling import freeze_bottom_encoder_layers


class CommonMetricsTest(unittest.TestCase):
    def test_arxiv_month(self):
        self.assertEqual(arxiv_month("2501.12345v2"), (2025, 1))
        self.assertEqual(arxiv_month("hep-th/9901001"), (1999, 1))

    def test_unique_top_k(self):
        self.assertEqual(unique_top_k(["a", "a", "x", "b"], {"a", "b"}, 2), ["a", "b"])

    def test_chronological_split_keeps_months_intact(self):
        rows = [
            {"query_id": "a", "published_at": "2025-01"},
            {"query_id": "b", "published_at": "2025-02"},
            {"query_id": "c", "published_at": "2025-03"},
            {"query_id": "d", "published_at": "2025-03"},
        ]
        train, dev, cutoff = chronological_split(rows, 0.25)
        self.assertEqual(cutoff, (2025, 3))
        self.assertEqual({row["query_id"] for row in train}, {"a", "b"})
        self.assertEqual({row["query_id"] for row in dev}, {"c", "d"})

    def test_metrics(self):
        result = metrics_at_k(["a", "x", "b"], ["a", "b", "c"], k=3)
        self.assertEqual(result["hits_at_3"], 2.0)
        self.assertAlmostEqual(result["precision_at_3"], 2 / 3)
        self.assertAlmostEqual(result["recall_at_3"], 2 / 3)
        self.assertEqual(result["mrr_at_3"], 1.0)

    def test_freeze_direct_encoder_layers(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Module()
        model.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])

        freeze_bottom_encoder_layers(model, 2)

        self.assertFalse(next(model.encoder.layer[0].parameters()).requires_grad)
        self.assertFalse(next(model.encoder.layer[1].parameters()).requires_grad)
        self.assertTrue(next(model.encoder.layer[2].parameters()).requires_grad)


if __name__ == "__main__":
    unittest.main()
