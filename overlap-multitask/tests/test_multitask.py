"""Baseline 호환성과 compact latent multitask 계약을 검증한다."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from mnist_overlap.config import (
    CHECKPOINT_DIR as BASELINE_CHECKPOINT_DIR,
    CLASS_COUNT,
    COMPOSITION_MODE,
)
from mnist_overlap.data import (
    RECONSTRUCTION_SIZE,
    ControlledOverlapMnistDataset,
    render_overlap_sample,
)
from mnist_overlap.model import MnistONet
from mnist_overlap.multitask.config import (
    CHECKPOINT_DIR as MULTITASK_CHECKPOINT_DIR,
    ReconstructionConfig,
    load_multitask_config,
    multitask_config_fingerprint,
)
from mnist_overlap.multitask.evaluation import crop_source_images
from mnist_overlap.multitask.losses import (
    foreground_dice_per_sample,
    source_reconstruction_loss,
)
from mnist_overlap.multitask.model import (
    CLASS_LATENT_DIMENSION,
    ClassLatentEncoder,
    MultitaskMnistONet,
    SharedReconstructionDecoder,
)
from mnist_overlap.multitask.training import (
    load_checkpoint,
    select_reconstruction_loss_weight,
)
from mnist_overlap.runtime import set_random_seed


class BaselineCompatibilityTest(unittest.TestCase):
    """기존 LeNet 구조와 완료 checkpoint 호환성을 검사한다."""

    def test_split_forward_matches_original_sequential_forward(self) -> None:
        model = MnistONet().eval()
        images = torch.rand(2, 1, 76, 76)
        with torch.no_grad():
            expected_logits = model.layers(images)
            actual_logits = model(images)
        self.assertTrue(torch.equal(expected_logits, actual_logits))

    def test_existing_seed_zero_checkpoint_keeps_layer_keys(self) -> None:
        checkpoint_path = BASELINE_CHECKPOINT_DIR / "seed_0.pt"
        if not checkpoint_path.exists():
            self.skipTest("완료된 baseline seed 0 checkpoint가 없습니다.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = MnistONet()
        model.load_state_dict(checkpoint["model_state_dict"])
        self.assertIn("layers.0.weight", checkpoint["model_state_dict"])

    def test_baseline_and_multitask_output_paths_are_isolated(self) -> None:
        self.assertNotEqual(BASELINE_CHECKPOINT_DIR, MULTITASK_CHECKPOINT_DIR)
        self.assertNotEqual(BASELINE_CHECKPOINT_DIR.parent, MULTITASK_CHECKPOINT_DIR.parent)


class DatasetContractTest(unittest.TestCase):
    """Source image 반환 옵션이 기존 sample 계약을 보존하는지 검사한다."""

    def test_overlap_composition_uses_clipped_sum(self) -> None:
        source_image_first = torch.tensor([[1.0, 0.2], [0.0, 0.0]])
        source_image_second = torch.tensor([[0.5, 0.3], [0.0, 0.0]])
        actual = render_overlap_sample(
            source_image_first,
            source_image_second,
            (0, 0),
            (0, 0),
            canvas_size=2,
        )
        expected = torch.tensor([[1.0, 0.5], [0.0, 0.0]])
        self.assertEqual(COMPOSITION_MODE, "clipped_sum")
        self.assertTrue(torch.equal(actual, expected))

    def test_source_images_are_optional_and_have_expected_shape(self) -> None:
        baseline_sample = ControlledOverlapMnistDataset("train")[0]
        multitask_sample = ControlledOverlapMnistDataset(
            "train", include_source_images=True
        )[0]
        self.assertNotIn("source_images", baseline_sample)
        self.assertEqual(tuple(multitask_sample["source_images"].shape), (2, 28, 28))
        self.assertEqual(
            tuple(multitask_sample["reconstruction_targets"].shape),
            (2, RECONSTRUCTION_SIZE, RECONSTRUCTION_SIZE),
        )
        self.assertEqual(tuple(multitask_sample["source_offsets"].shape), (2, 2))
        self.assertTrue(torch.equal(baseline_sample["image"], multitask_sample["image"]))
        self.assertTrue(torch.equal(baseline_sample["label"], multitask_sample["label"]))

        recovered_sources = crop_source_images(
            multitask_sample["reconstruction_targets"].unsqueeze(0),
            multitask_sample["source_offsets"].unsqueeze(0),
            28,
        )[0]
        self.assertTrue(torch.equal(recovered_sources, multitask_sample["source_images"]))


class MultitaskModelTest(unittest.TestCase):
    """Compact latent shape·초기화와 두 loss의 gradient 경로를 검사한다."""

    def test_class_latent_and_output_shapes(self) -> None:
        images = torch.rand(2, 1, 76, 76)
        classes = torch.tensor([[3, 8], [1, 7]])
        model = MultitaskMnistONet()
        bottleneck = model.classifier.encode(images)
        class_latents = model.reconstruction_head.latent_encoder(bottleneck)
        output = model(images, classes)

        self.assertEqual(
            tuple(class_latents.shape),
            (2, CLASS_COUNT, CLASS_LATENT_DIMENSION),
        )
        self.assertEqual(tuple(output.logits.shape), (2, CLASS_COUNT))
        self.assertEqual(
            tuple(output.reconstruction_logits.shape),
            (2, 2, RECONSTRUCTION_SIZE, RECONSTRUCTION_SIZE),
        )
        self.assertTrue(torch.equal(output.reconstruction_classes, classes))
        probabilities = torch.sigmoid(output.reconstruction_logits).detach()
        self.assertGreaterEqual(float(probabilities.min()), 0.0)
        self.assertLessEqual(float(probabilities.max()), 1.0)

    def test_decoder_is_shared_mlp_without_skip_convolutions(self) -> None:
        model = MultitaskMnistONet()
        decoder = model.reconstruction_head.decoder
        self.assertIsInstance(decoder, SharedReconstructionDecoder)
        self.assertTrue(any(
            isinstance(module, torch.nn.Linear) for module in decoder.modules()
        ))
        self.assertFalse(any(
            isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
            for module in model.reconstruction_head.modules()
        ))

    def test_default_reconstruction_classes_are_classifier_top_two(self) -> None:
        model = MultitaskMnistONet()
        output = model(torch.rand(2, 1, 76, 76))
        expected_classes = torch.topk(output.logits, k=2, dim=1).indices
        self.assertTrue(torch.equal(output.reconstruction_classes, expected_classes))

    def test_invalid_reconstruction_classes_are_rejected(self) -> None:
        model = MultitaskMnistONet()
        images = torch.rand(2, 1, 76, 76)
        invalid_values = (
            torch.tensor([[3, 3], [1, 7]]),
            torch.tensor([[3, 10], [1, 7]]),
            torch.tensor([3, 8]),
            torch.tensor([[3.0, 8.0], [1.0, 7.0]]),
        )
        for invalid_classes in invalid_values:
            with self.subTest(classes=invalid_classes):
                with self.assertRaises(ValueError):
                    model(images, invalid_classes)

    def test_classifier_initialization_matches_baseline_for_same_seed(self) -> None:
        set_random_seed(7)
        baseline = MnistONet()
        baseline_state = {
            name: parameter.detach().clone()
            for name, parameter in baseline.state_dict().items()
        }
        set_random_seed(7)
        multitask = MultitaskMnistONet()
        for name, parameter in multitask.classifier.state_dict().items():
            self.assertTrue(torch.equal(baseline_state[name], parameter))

    def test_reconstruction_and_classification_gradient_routes(self) -> None:
        model = MultitaskMnistONet()
        images = torch.rand(2, 1, 76, 76)
        classes = torch.tensor([[3, 8], [3, 8]])
        reconstruction_targets = torch.zeros(
            2, 2, RECONSTRUCTION_SIZE, RECONSTRUCTION_SIZE
        )
        reconstruction_targets[:, 0, 4:20, 5:18] = 1.0
        reconstruction_targets[:, 1, 30:48, 32:50] = 1.0
        output = model(images, classes)
        reconstruction_loss = source_reconstruction_loss(
            output.reconstruction_logits,
            reconstruction_targets,
        ).loss
        reconstruction_loss.backward()
        self.assertIsNotNone(model.classifier.layers[0].weight.grad)
        self.assertIsNone(model.classifier.layers[7].weight.grad)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.reconstruction_head.parameters()
        ))

        model.zero_grad(set_to_none=True)
        output = model(images, classes)
        labels = torch.zeros(2, CLASS_COUNT)
        labels[:, [3, 8]] = 1.0
        classification_loss = torch.nn.BCEWithLogitsLoss()(output.logits, labels)
        classification_loss.backward()
        self.assertIsNotNone(model.classifier.layers[0].weight.grad)
        self.assertIsNotNone(model.classifier.layers[7].weight.grad)
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.reconstruction_head.parameters()
        ))


class ReconstructionLossTest(unittest.TestCase):
    """Source BCE·Dice가 foreground와 source identity를 학습하는지 검사한다."""

    def test_source_loss_penalizes_blank_mixture_and_swap(self) -> None:
        reconstruction_targets = torch.zeros(
            2, 2, RECONSTRUCTION_SIZE, RECONSTRUCTION_SIZE
        )
        reconstruction_targets[:, 0, 4:12, 5:13] = 1.0
        reconstruction_targets[:, 1, 15:23, 16:24] = 1.0
        perfect_logits = torch.where(
            reconstruction_targets > 0,
            torch.full_like(reconstruction_targets, 12.0),
            torch.full_like(reconstruction_targets, -12.0),
        )
        blank_logits = torch.full_like(reconstruction_targets, -12.0)
        mixed_foreground = torch.maximum(
            reconstruction_targets[:, 0],
            reconstruction_targets[:, 1],
        )
        mixed_targets = torch.stack((mixed_foreground, mixed_foreground), dim=1)
        mixed_logits = torch.where(
            mixed_targets > 0,
            torch.full_like(mixed_targets, 12.0),
            torch.full_like(mixed_targets, -12.0),
        )
        swapped_logits = perfect_logits[:, [1, 0]]

        perfect = source_reconstruction_loss(perfect_logits, reconstruction_targets)
        blank = source_reconstruction_loss(blank_logits, reconstruction_targets)
        mixed = source_reconstruction_loss(mixed_logits, reconstruction_targets)
        swapped = source_reconstruction_loss(swapped_logits, reconstruction_targets)
        self.assertGreater(float(blank.loss), float(perfect.loss))
        self.assertGreater(float(mixed.loss), float(perfect.loss))
        self.assertGreater(float(swapped.loss), float(perfect.loss))

        perfect_dice = foreground_dice_per_sample(
            reconstruction_targets,
            reconstruction_targets,
        )
        blank_dice = foreground_dice_per_sample(
            torch.zeros_like(reconstruction_targets),
            reconstruction_targets,
        )
        self.assertTrue(torch.allclose(perfect_dice, torch.ones_like(perfect_dice)))
        self.assertTrue(torch.all(blank_dice < perfect_dice))

    def test_pilot_tie_break_prefers_smallest_weight_within_tolerance(self) -> None:
        candidates = [
            {"reconstruction_loss_weight": 0.05, "best_validation_exact_match": 0.8005},
            {"reconstruction_loss_weight": 0.1, "best_validation_exact_match": 0.8010},
            {"reconstruction_loss_weight": 0.2, "best_validation_exact_match": 0.7990},
        ]
        self.assertEqual(select_reconstruction_loss_weight(candidates, 0.001), 0.05)


class MultitaskCheckpointTest(unittest.TestCase):
    """Checkpoint 완료 상태와 config fingerprint 검사를 확인한다."""

    def test_incomplete_and_incompatible_checkpoints_are_rejected(self) -> None:
        config = load_multitask_config()
        reconstruction_loss_weight = 0.1
        model = MultitaskMnistONet()
        checkpoint = {
            "config_fingerprint": multitask_config_fingerprint(
                config, reconstruction_loss_weight
            ),
            "training_complete": False,
            "classifier_state_dict": model.classifier.state_dict(),
            "reconstruction_head_state_dict": model.reconstruction_head.state_dict(),
        }

        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "정상 종료되지 않은"):
                load_checkpoint(
                    MultitaskMnistONet(),
                    checkpoint_path,
                    torch.device("cpu"),
                    config,
                    reconstruction_loss_weight,
                )

            checkpoint["training_complete"] = True
            checkpoint["config_fingerprint"] = "incompatible"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "현재 config와 다른"):
                load_checkpoint(
                    MultitaskMnistONet(),
                    checkpoint_path,
                    torch.device("cpu"),
                    config,
                    reconstruction_loss_weight,
                )

    def test_pilot_contract_rejects_unplanned_values(self) -> None:
        with self.assertRaises(ValueError):
            ReconstructionConfig(1, (0.05, 0.1, 0.2))
        with self.assertRaises(ValueError):
            ReconstructionConfig(0, (0.1, 0.2, 0.3))


if __name__ == "__main__":
    unittest.main()
