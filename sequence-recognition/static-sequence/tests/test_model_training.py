import copy
import unittest

import torch
from torch.utils.data import DataLoader, Subset

from static_sequence_core.config import load_config
from static_sequence_core.dataset import create_dataset
from static_sequence_core.engine import sequence_loss
from static_sequence_core.model import StaticSequenceRecognizer


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()
        torch.manual_seed(0)

    def test_forward_shape_is_finite_and_backward_works(self) -> None:
        model = StaticSequenceRecognizer(self.cfg)
        images = torch.zeros(2, 1, 32, 96, dtype=torch.uint8)
        labels = torch.tensor([[0, 0, 7], [3, 8, 1]])
        logits = model(images)
        self.assertEqual(logits.shape, (2, 3, 10))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        loss = sequence_loss(logits, labels)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_tiny_subset_can_overfit(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["dataset"]["train_samples"] = 8
        dataset = create_dataset(cfg, train=True)
        loader = DataLoader(Subset(dataset, range(8)), batch_size=8, shuffle=False)
        images, labels = next(iter(loader))
        model = StaticSequenceRecognizer(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        with torch.no_grad():
            initial_loss = float(sequence_loss(model(images), labels).item())
        for _ in range(80):
            optimizer.zero_grad()
            loss = sequence_loss(model(images), labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final_logits = model(images)
            final_loss = float(sequence_loss(final_logits, labels).item())
            exact_match = float(final_logits.argmax(dim=2).eq(labels).all(dim=1).float().mean())

        self.assertLess(final_loss, initial_loss * 0.1)
        self.assertGreaterEqual(exact_match, 0.875)


if __name__ == "__main__":
    unittest.main()
