import unittest

import torch
from torchvision.datasets import MNIST

from static_sequence_core.codec import digits_to_string, string_to_digits
from static_sequence_core.config import DATA_DIR
from static_sequence_core.dataset import StaticSequenceDataset


class CodecTest(unittest.TestCase):
    def test_round_trip_preserves_leading_zero(self) -> None:
        for text in ("007", "010", "381"):
            self.assertEqual(digits_to_string(string_to_digits(text)), text)

    def test_rejects_invalid_sequence(self) -> None:
        with self.assertRaises(ValueError):
            digits_to_string([1, 2])
        with self.assertRaises(ValueError):
            string_to_digits("12a")


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = MNIST(DATA_DIR, train=False, download=True)

    def make_dataset(self, seed: int) -> StaticSequenceDataset:
        return StaticSequenceDataset(self.source, 8, 3, 32, seed)

    def test_shape_dtype_range_and_padding(self) -> None:
        image, labels = self.make_dataset(7)[0]
        self.assertEqual(image.shape, (1, 32, 96))
        self.assertEqual(image.dtype, torch.uint8)
        self.assertEqual(labels.shape, (3,))
        self.assertEqual(labels.dtype, torch.int64)
        self.assertTrue(bool(torch.all((labels >= 0) & (labels <= 9))))
        self.assertEqual(int(image[:, :2, :].sum()), 0)
        self.assertEqual(int(image[:, 30:, :].sum()), 0)
        for boundary in (0, 32, 64, 96):
            if boundary < 96:
                self.assertEqual(int(image[:, :, boundary:boundary + 2].sum()), 0)

    def test_same_seed_and_index_are_deterministic(self) -> None:
        first_image, first_labels = self.make_dataset(11)[3]
        second_image, second_labels = self.make_dataset(11)[3]
        self.assertTrue(torch.equal(first_image, second_image))
        self.assertTrue(torch.equal(first_labels, second_labels))


if __name__ == "__main__":
    unittest.main()
