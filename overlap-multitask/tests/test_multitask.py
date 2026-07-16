"""Baseline 호환성과 multitask 모델·loss·gradient 계약을 검증한다."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from mnist_overlap.config import CHECKPOINT_DIR as BASELINE_CHECKPOINT_DIR
from mnist_overlap.data import (
    RECONSTRUCTION_SIZE,
    ControlledOverlapMnistDataset,
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
    permutation_invariant_reconstruction_loss,
)
from mnist_overlap.multitask.model import MultitaskMnistONet
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
    """Decoder shape·초기화와 두 loss의 gradient 경로를 검사한다."""

    def test_encoder_skip_shapes_match_unet_contract(self) -> None:
        features = MnistONet().encode_with_skips(torch.rand(2, 1, 76, 76))
        self.assertEqual(tuple(features[0].shape), (2, 6, 72, 72))
        self.assertEqual(tuple(features[1].shape), (2, 16, 32, 32))
        self.assertEqual(tuple(features[2].shape), (2, 16, 16, 16))

    def test_output_shape_and_range(self) -> None:
        output = MultitaskMnistONet()(torch.rand(2, 1, 76, 76))
        self.assertEqual(tuple(output.logits.shape), (2, 10))
        self.assertEqual(
            tuple(output.reconstructions.shape),
            (2, 2, RECONSTRUCTION_SIZE, RECONSTRUCTION_SIZE),
        )
        self.assertGreaterEqual(float(output.reconstructions.min()), 0.0)
        self.assertLessEqual(float(output.reconstructions.max()), 1.0)

    def test_decoder_is_fully_convolutional(self) -> None:
        decoder = MultitaskMnistONet().decoder
        self.assertFalse(any(
            isinstance(module, (torch.nn.Flatten, torch.nn.Linear))
            for module in decoder.modules()
        ))

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
        reconstruction_targets = torch.rand(
            2, 2, RECONSTRUCTION_SIZE, RECONSTRUCTION_SIZE
        )
        output = model(images)
        reconstruction_loss = permutation_invariant_reconstruction_loss(
            output.reconstructions, reconstruction_targets
        ).loss
        reconstruction_loss.backward()
        self.assertIsNotNone(model.classifier.layers[0].weight.grad)
        self.assertIsNone(model.classifier.layers[7].weight.grad)
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.decoder.parameters()
        ))

        model.zero_grad(set_to_none=True)
        output = model(images)
        labels = torch.zeros(2, 10)
        labels[:, :2] = 1.0
        classification_loss = torch.nn.BCEWithLogitsLoss()(output.logits, labels)
        classification_loss.backward()
        self.assertIsNotNone(model.classifier.layers[0].weight.grad)
        self.assertIsNotNone(model.classifier.layers[7].weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.decoder.parameters()))


class ReconstructionLossTest(unittest.TestCase):
    """Balanced PIT loss의 순서 불변성과 blank 억제 성질을 검사한다."""

    def test_pit_is_invariant_to_output_channel_swap(self) -> None:
        reconstruction_targets = torch.zeros(
            2, 2, RECONSTRUCTION_SIZE, RECONSTRUCTION_SIZE
        )
        reconstruction_targets[:, 0, 4:12, 5:13] = 1.0
        reconstruction_targets[:, 1, 15:23, 16:24] = 0.8
        direct = permutation_invariant_reconstruction_loss(
            reconstruction_targets, reconstruction_targets
        )
        swapped = permutation_invariant_reconstruction_loss(
            reconstruction_targets[:, [1, 0]], reconstruction_targets
        )
        blank = permutation_invariant_reconstruction_loss(
            torch.zeros_like(reconstruction_targets), reconstruction_targets
        )
        self.assertAlmostEqual(float(direct.loss), float(swapped.loss), places=7)
        self.assertGreater(float(blank.loss), float(direct.loss))
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
            "decoder_state_dict": model.decoder.state_dict(),
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
