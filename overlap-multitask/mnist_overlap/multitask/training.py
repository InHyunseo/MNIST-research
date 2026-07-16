"""Multitask pilot, 공동학습, early stopping과 checkpoint 관리를 담당한다."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data import ControlledOverlapMnistDataset
from ..metrics import exact_match_per_sample
from ..runtime import create_train_loader, save_checkpoint_atomically, set_random_seed
from .config import (
    CHECKPOINT_DIR,
    PILOT_SELECTION_PATH,
    TRAINING_LOG_DIR,
    MultitaskConfig,
    create_multitask_directories,
    multitask_config_fingerprint,
    pilot_weight_directory,
)
from .losses import permutation_invariant_reconstruction_loss
from .model import MultitaskMnistONet


@dataclass(frozen=True)
class MultitaskEpochResult:
    """한 epoch의 sample-weighted loss 세 항목과 분류 exact-match."""

    total_loss: float
    classification_loss: float
    reconstruction_loss: float
    exact_match: float


@dataclass(frozen=True)
class MultitaskTrainingResult:
    """한 seed 공동학습의 best epoch와 생성 산출물 경로."""

    seed: int
    reconstruction_loss_weight: float
    best_epoch: int
    epochs_run: int
    best_validation_exact_match: float
    checkpoint_path: Path
    history_path: Path


def train_one_epoch(
    model: MultitaskMnistONet,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    classification_loss_function: nn.Module,
    reconstruction_loss_weight: float,
    device: torch.device,
) -> MultitaskEpochResult:
    """DataLoader 전체를 순회해 분류·복원 loss로 모든 multitask parameter를 갱신한다."""
    model.train()
    totals = _empty_epoch_totals()

    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        source_images = batch["source_images"].to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        classification_loss = classification_loss_function(output.logits, labels)
        pit_result = permutation_invariant_reconstruction_loss(
            output.reconstructions, source_images
        )
        total_loss = classification_loss + reconstruction_loss_weight * pit_result.loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"유한하지 않은 training loss입니다: {total_loss.item()}")
        total_loss.backward()
        optimizer.step()

        _accumulate_epoch_totals(
            totals,
            images.shape[0],
            total_loss,
            classification_loss,
            pit_result.loss,
            output.logits.detach(),
            labels,
        )

    return _finalize_epoch_totals(totals)


@torch.no_grad()
def evaluate_validation(
    model: MultitaskMnistONet,
    data_loader: DataLoader,
    classification_loss_function: nn.Module,
    reconstruction_loss_weight: float,
    device: torch.device,
) -> MultitaskEpochResult:
    """Parameter를 변경하지 않고 validation 분류·복원 성능을 계산한다."""
    model.eval()
    totals = _empty_epoch_totals()

    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        source_images = batch["source_images"].to(device)
        output = model(images)
        classification_loss = classification_loss_function(output.logits, labels)
        pit_result = permutation_invariant_reconstruction_loss(
            output.reconstructions, source_images
        )
        total_loss = classification_loss + reconstruction_loss_weight * pit_result.loss

        _accumulate_epoch_totals(
            totals,
            images.shape[0],
            total_loss,
            classification_loss,
            pit_result.loss,
            output.logits,
            labels,
        )

    return _finalize_epoch_totals(totals)


def train_model(
    model: MultitaskMnistONet,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: MultitaskConfig,
    seed: int,
    reconstruction_loss_weight: float,
    checkpoint_path: Path,
    history_path: Path,
    device: torch.device,
) -> MultitaskTrainingResult:
    """한 seed를 early stopping까지 공동학습하고 validation best checkpoint를 저장한다."""
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.baseline.training.learning_rate,
    )
    classification_loss_function = nn.BCEWithLogitsLoss()
    best_exact_match = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    experiment_fingerprint = multitask_config_fingerprint(
        config, reconstruction_loss_weight
    )

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow([
            "epoch",
            "train_total_loss",
            "train_classification_loss",
            "train_reconstruction_loss",
            "train_exact_match",
            "validation_total_loss",
            "validation_classification_loss",
            "validation_reconstruction_loss",
            "validation_exact_match",
        ])

        for epoch in range(1, config.baseline.training.maximum_epochs + 1):
            train_result = train_one_epoch(
                model,
                train_loader,
                optimizer,
                classification_loss_function,
                reconstruction_loss_weight,
                device,
            )
            validation_result = evaluate_validation(
                model,
                validation_loader,
                classification_loss_function,
                reconstruction_loss_weight,
                device,
            )
            writer.writerow([
                epoch,
                f"{train_result.total_loss:.8f}",
                f"{train_result.classification_loss:.8f}",
                f"{train_result.reconstruction_loss:.8f}",
                f"{train_result.exact_match:.8f}",
                f"{validation_result.total_loss:.8f}",
                f"{validation_result.classification_loss:.8f}",
                f"{validation_result.reconstruction_loss:.8f}",
                f"{validation_result.exact_match:.8f}",
            ])
            history_file.flush()
            print(
                f"seed={seed} lambda={reconstruction_loss_weight:g} epoch={epoch} "
                f"train_cls={train_result.classification_loss:.4f} "
                f"train_rec={train_result.reconstruction_loss:.4f} "
                f"validation_exact={validation_result.exact_match:.4f}"
            )

            improved = (
                validation_result.exact_match
                > best_exact_match + config.baseline.training.early_stopping_minimum_delta
            )
            if improved:
                best_exact_match = validation_result.exact_match
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint_atomically({
                    "seed": seed,
                    "reconstruction_loss_weight": reconstruction_loss_weight,
                    "best_epoch": epoch,
                    "validation_exact_match": best_exact_match,
                    "config_fingerprint": experiment_fingerprint,
                    "training_complete": False,
                    "classifier_state_dict": model.classifier.state_dict(),
                    "decoder_state_dict": model.decoder.state_dict(),
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= config.baseline.training.early_stopping_patience
            ):
                break

    if best_epoch == 0:
        raise RuntimeError("Multitask 학습에서 checkpoint를 생성하지 못했습니다.")

    epochs_run = epoch
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    checkpoint["epochs_run"] = epochs_run
    checkpoint["training_complete"] = True
    save_checkpoint_atomically(checkpoint, checkpoint_path)
    return MultitaskTrainingResult(
        seed,
        reconstruction_loss_weight,
        best_epoch,
        epochs_run,
        best_exact_match,
        checkpoint_path,
        history_path,
    )


def run_pilot(config: MultitaskConfig, device: torch.device) -> float:
    """세 가중치를 pilot seed로 학습하고 validation 성능으로 최종 λ를 선택한다."""
    create_multitask_directories()
    saved_selection = _load_compatible_pilot_selection(config)
    if saved_selection is not None:
        selected_weight = float(saved_selection["selected_reconstruction_loss_weight"])
        print(f"기존 pilot 선택을 사용합니다: lambda={selected_weight:g}")
        return selected_weight

    train_dataset = ControlledOverlapMnistDataset(
        "train", include_source_images=True
    )
    validation_dataset = ControlledOverlapMnistDataset(
        "validation", include_source_images=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.baseline.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    candidate_results = []

    for reconstruction_loss_weight in config.reconstruction.loss_weight_candidates:
        candidate_directory = pilot_weight_directory(reconstruction_loss_weight)
        checkpoint_path = candidate_directory / f"seed_{config.reconstruction.pilot_seed}.pt"
        history_path = candidate_directory / f"seed_{config.reconstruction.pilot_seed}.csv"
        result = _train_or_reuse(
            config,
            config.reconstruction.pilot_seed,
            reconstruction_loss_weight,
            checkpoint_path,
            history_path,
            train_dataset,
            validation_loader,
            device,
        )
        candidate_results.append({
            "reconstruction_loss_weight": reconstruction_loss_weight,
            "best_validation_exact_match": result.best_validation_exact_match,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
        })

    tolerance = config.baseline.training.early_stopping_minimum_delta
    selected_weight = select_reconstruction_loss_weight(candidate_results, tolerance)
    selection = {
        "config_fingerprint": multitask_config_fingerprint(config),
        "pilot_seed": config.reconstruction.pilot_seed,
        "selection_tolerance": tolerance,
        "selected_reconstruction_loss_weight": selected_weight,
        "candidates": candidate_results,
    }
    _write_json_atomically(selection, PILOT_SELECTION_PATH)
    print(f"Pilot 선택 완료: lambda={selected_weight:g}")
    return selected_weight


def load_selected_reconstruction_weight(config: MultitaskConfig) -> float:
    """현재 config와 호환되는 pilot 선택 결과에서 최종 λ를 읽는다."""
    selection = _load_compatible_pilot_selection(config)
    if selection is None:
        raise FileNotFoundError("호환되는 pilot selection.json이 없습니다.")
    return float(selection["selected_reconstruction_loss_weight"])


def select_reconstruction_loss_weight(
    candidate_results: list[dict[str, float | int]],
    tolerance: float,
) -> float:
    """최고 validation exact와 tolerance 이내인 후보 중 가장 작은 λ를 선택한다."""
    if not candidate_results:
        raise ValueError("가중치를 선택할 pilot 결과가 없습니다.")
    maximum_exact_match = max(
        float(row["best_validation_exact_match"]) for row in candidate_results
    )
    eligible_weights = [
        float(row["reconstruction_loss_weight"])
        for row in candidate_results
        if maximum_exact_match - float(row["best_validation_exact_match"]) <= tolerance
    ]
    return min(eligible_weights)


def train_all_seeds(
    config: MultitaskConfig,
    reconstruction_loss_weight: float,
    device: torch.device,
) -> list[MultitaskTrainingResult]:
    """선택된 λ로 전체 seed를 공동학습하고 완료된 checkpoint는 재사용한다."""
    create_multitask_directories()
    train_dataset = ControlledOverlapMnistDataset(
        "train", include_source_images=True
    )
    validation_dataset = ControlledOverlapMnistDataset(
        "validation", include_source_images=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.baseline.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    results = []

    for seed in config.baseline.training.seeds:
        result = _train_or_reuse(
            config,
            seed,
            reconstruction_loss_weight,
            CHECKPOINT_DIR / f"seed_{seed}.pt",
            TRAINING_LOG_DIR / f"seed_{seed}.csv",
            train_dataset,
            validation_loader,
            device,
        )
        results.append(result)
    return results


def load_checkpoint(
    model: MultitaskMnistONet,
    checkpoint_path: Path,
    device: torch.device,
    config: MultitaskConfig,
    reconstruction_loss_weight: float,
) -> dict[str, Any]:
    """호환되는 완료 checkpoint를 classifier와 decoder에 복원한다."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected_fingerprint = multitask_config_fingerprint(
        config, reconstruction_loss_weight
    )
    if checkpoint.get("config_fingerprint") != expected_fingerprint:
        raise ValueError(f"현재 config와 다른 multitask checkpoint입니다: {checkpoint_path}")
    if checkpoint.get("training_complete") is not True:
        raise ValueError(f"정상 종료되지 않은 multitask checkpoint입니다: {checkpoint_path}")
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    model.to(device)
    return checkpoint


def _train_or_reuse(
    config: MultitaskConfig,
    seed: int,
    reconstruction_loss_weight: float,
    checkpoint_path: Path,
    history_path: Path,
    train_dataset: ControlledOverlapMnistDataset,
    validation_loader: DataLoader,
    device: torch.device,
) -> MultitaskTrainingResult:
    """호환 checkpoint가 있으면 결과를 복원하고 없으면 같은 seed로 새로 학습한다."""
    expected_fingerprint = multitask_config_fingerprint(
        config, reconstruction_loss_weight
    )
    if _checkpoint_matches(checkpoint_path, expected_fingerprint):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print(f"기존 multitask checkpoint를 사용합니다: {checkpoint_path}")
        return MultitaskTrainingResult(
            seed,
            reconstruction_loss_weight,
            int(checkpoint["best_epoch"]),
            int(checkpoint["epochs_run"]),
            float(checkpoint["validation_exact_match"]),
            checkpoint_path,
            history_path,
        )

    for artifact_path in (
        checkpoint_path,
        checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp"),
        history_path,
    ):
        artifact_path.unlink(missing_ok=True)

    set_random_seed(seed)
    model = MultitaskMnistONet()
    train_loader = create_train_loader(
        train_dataset, config.baseline.training, seed
    )
    return train_model(
        model,
        train_loader,
        validation_loader,
        config,
        seed,
        reconstruction_loss_weight,
        checkpoint_path,
        history_path,
        device,
    )


def _checkpoint_matches(checkpoint_path: Path, expected_fingerprint: str) -> bool:
    """Checkpoint가 기대 config로 정상 완료됐는지 확인한다."""
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return (
        checkpoint.get("config_fingerprint") == expected_fingerprint
        and checkpoint.get("training_complete") is True
    )


def _load_compatible_pilot_selection(config: MultitaskConfig) -> dict[str, Any] | None:
    """현재 config 지문과 후보 집합에 맞는 pilot 선택 결과만 반환한다."""
    if not PILOT_SELECTION_PATH.exists():
        return None
    selection = json.loads(PILOT_SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("config_fingerprint") != multitask_config_fingerprint(config):
        return None
    selected_weight = float(selection.get("selected_reconstruction_loss_weight", -1.0))
    if selected_weight not in config.reconstruction.loss_weight_candidates:
        return None
    return selection


def _write_json_atomically(payload: dict[str, Any], path: Path) -> None:
    """JSON을 임시 파일에 쓴 뒤 목표 경로로 원자 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _empty_epoch_totals() -> dict[str, float]:
    """한 epoch 동안 누적할 sample-weighted 합계를 초기화한다."""
    return {
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "reconstruction_loss": 0.0,
        "correct": 0.0,
        "samples": 0.0,
    }


def _accumulate_epoch_totals(
    totals: dict[str, float],
    batch_size: int,
    total_loss: torch.Tensor,
    classification_loss: torch.Tensor,
    reconstruction_loss: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """한 batch의 loss와 정답 수를 epoch 누적값에 더한다."""
    totals["total_loss"] += float(total_loss.item()) * batch_size
    totals["classification_loss"] += float(classification_loss.item()) * batch_size
    totals["reconstruction_loss"] += float(reconstruction_loss.item()) * batch_size
    totals["correct"] += float(exact_match_per_sample(logits, labels).sum().item())
    totals["samples"] += batch_size


def _finalize_epoch_totals(totals: dict[str, float]) -> MultitaskEpochResult:
    """누적 합계를 sample 수로 나눠 한 epoch 평균을 만든다."""
    sample_count = totals["samples"]
    if sample_count <= 0:
        raise ValueError("비어 있는 DataLoader에서는 epoch 결과를 계산할 수 없습니다.")
    return MultitaskEpochResult(
        totals["total_loss"] / sample_count,
        totals["classification_loss"] / sample_count,
        totals["reconstruction_loss"] / sample_count,
        totals["correct"] / sample_count,
    )
