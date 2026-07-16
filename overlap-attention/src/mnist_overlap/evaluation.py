"""Checkpoint 평가, run cache 관리, 집계 CSV 생성.

Run별 prediction·attention metric을 fingerprint 붙은 NPZ cache로 저장하고,
전체 run이 모이면 hierarchical bootstrap 등 최종 통계표를 `outputs/metrics/`에 쓴다.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .analysis import (
    collect_attention_metric_arrays,
    collect_prediction_arrays,
    count_parameters,
    create_seed_effect_rows,
    difference_in_differences,
    estimate_multiply_accumulate_operations,
    hierarchical_bootstrap_interval,
    paired_level_difference,
)
from .config import (
    ATTENTION_LOG_DIR,
    CHECKPOINT_DIR,
    CLASS_COUNT,
    DEFAULT_CONFIG_PATH,
    METRIC_LOG_DIR,
    PREDICTION_LOG_DIR,
    TRAINING_LOG_DIR,
    EvaluationConfig,
    ExperimentConfig,
    config_fingerprint,
    create_output_directories,
    evaluation_fingerprint,
    load_config,
    update_experiment_metadata,
)
from .data import ControlledOverlapMnistDataset
from .metrics import class_pair_accuracy, classification_metrics, finite_mean, sample_deviation
from .models import create_model
from .training import load_checkpoint, select_device, select_model_names, select_seeds


PREDICTION_CACHE_FIELDS = {
    "logits",
    "labels",
    "sample_id",
    "pair_id",
    "label_first",
    "label_second",
    "bounding_box_overlap_ratio",
    "pixel_overlap_ratio",
    "overlap_level",
    "cache_fingerprint",
}
ATTENTION_CACHE_FIELDS = {
    "sample_id",
    "pair_id",
    "overlap_level",
    "average_precision",
    "iou",
    "cross_selectivity",
    "normal_correct",
    "permuted_correct",
    "validation_thresholds",
    "validation_mean_iou",
    "selected_iou_threshold",
    "cache_fingerprint",
}


# -----------------------------------------------------------------------------
# 공개 실행 API
# -----------------------------------------------------------------------------


def analyze_saved_results(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    """저장된 run cache와 학습 log에서 모든 집계 CSV를 다시 생성한다."""
    config = load_config(config_path)
    create_output_directories()
    predictions_by_run = _load_all_prediction_caches(config, require_all=True)
    attention_by_run = _load_all_attention_caches(config, require_all=True)
    return _write_aggregate_results(predictions_by_run, attention_by_run, config)


def evaluate_models(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_name: str | None = None,
    seed: int | None = None,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> list[Path]:
    """선택한 checkpoint를 평가하고 유효한 run cache는 재사용한다.

    전체 run cache가 모이면 hierarchical 통계까지 함께 집계한다.
    """
    config = load_config(config_path)
    create_output_directories()
    model_names = select_model_names(config, model_name)
    seeds = select_seeds(config, seed)
    device = select_device(device_name)

    if overwrite:
        _remove_evaluation_caches(model_names, seeds)

    batch_size = config.evaluation.batch_size
    workers = config.train.data_loader_workers
    test_dataset = ControlledOverlapMnistDataset("test", config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    attention_loaders: tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ] | None = None
    generated_paths: list[Path] = []

    for selected_model_name in model_names:
        for selected_seed in seeds:
            checkpoint_path = (
                CHECKPOINT_DIR / f"{selected_model_name}_seed_{selected_seed}.pt"
            )
            model: torch.nn.Module | None = None
            prediction_path = _prediction_cache_path(
                selected_model_name,
                selected_seed,
            )
            predictions = _load_cache(
                prediction_path,
                config,
                PREDICTION_CACHE_FIELDS,
            )

            if predictions is None:
                model = _load_run_model(
                    selected_model_name,
                    selected_seed,
                    checkpoint_path,
                    config,
                    device,
                )
                predictions = collect_prediction_arrays(model, test_loader, device)
                _save_cache(prediction_path, predictions, config)
                print(
                    f"Prediction 계산 완료: model={selected_model_name}, "
                    f"seed={selected_seed}"
                )
            else:
                print(f"Prediction cache 사용: {prediction_path}")
            generated_paths.append(prediction_path)

            if selected_model_name not in ("shared_attention", "class_attention"):
                continue

            attention_path = _attention_cache_path(
                selected_model_name,
                selected_seed,
            )
            attention_cache = _load_cache(
                attention_path,
                config,
                ATTENTION_CACHE_FIELDS,
            )
            if attention_cache is None:
                if model is None:
                    model = _load_run_model(
                        selected_model_name,
                        selected_seed,
                        checkpoint_path,
                        config,
                        device,
                    )
                if attention_loaders is None:
                    attention_loaders = _create_attention_loaders(
                        config,
                        batch_size,
                        workers,
                    )
                attention_cache = _evaluate_attention_run(
                    model=model,
                    model_name=selected_model_name,
                    validation_loader=attention_loaders[0],
                    test_loader=attention_loaders[1],
                    plain_test_loader=test_loader,
                    evaluation_config=config.evaluation,
                    device=device,
                    normal_predictions=predictions,
                )
                _save_cache(attention_path, attention_cache, config)
                print(
                    f"Attention metric 계산 완료: model={selected_model_name}, "
                    f"seed={selected_seed}"
                )
            else:
                print(f"Attention cache 사용: {attention_path}")
            generated_paths.append(attention_path)

    predictions_by_run = _load_all_prediction_caches(config, require_all=False)
    attention_by_run = _load_all_attention_caches(config, require_all=False)
    generated_paths.extend(
        _write_aggregate_results(predictions_by_run, attention_by_run, config)
    )
    update_experiment_metadata(config, device_name)
    return list(dict.fromkeys(generated_paths))


# -----------------------------------------------------------------------------
# CSV I/O (reporting과 공유)
# -----------------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """UTF-8 CSV를 dictionary row 목록으로 읽는다. 없거나 비어 있으면 오류를 낸다."""
    if not path.exists():
        raise FileNotFoundError(f"필요한 metric 파일이 없습니다: {path}")
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"Metric 파일이 비어 있습니다: {path}")
    return rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Dictionary row 목록을 UTF-8 CSV로 저장한다.

    부분 실행 상태를 표현하기 위해 빈 row 목록은 빈 파일로 기록한다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Run별 inference와 cache
# -----------------------------------------------------------------------------


def _load_run_model(
    model_name: str,
    seed: int,
    checkpoint_path: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> torch.nn.Module:
    """정상 완료된 checkpoint를 검증해 평가용 모델로 복원한다."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint가 없습니다: {checkpoint_path}")
    model = create_model(model_name, config)
    load_checkpoint(model, checkpoint_path, device, config)
    return model


def _evaluate_attention_run(
    model: torch.nn.Module,
    model_name: str,
    validation_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    plain_test_loader: torch.utils.data.DataLoader,
    evaluation_config: EvaluationConfig,
    device: torch.device,
    normal_predictions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validation IoU로 threshold를 고른 뒤 test sample별 attention metric을 계산한다.

    Class attention 모델은 permutation prediction도 함께 계산한다.
    """
    thresholds = _attention_thresholds(evaluation_config)
    minimum_pixels = evaluation_config.minimum_exclusive_pixels
    validation_metrics = collect_attention_metric_arrays(
        model,
        validation_loader,
        device,
        thresholds,
        minimum_pixels,
        compute_alignment=False,
    )
    validation_mean_iou = np.nanmean(
        validation_metrics["iou_by_threshold"],
        axis=1,
    )
    best_threshold_index = int(np.nanargmax(validation_mean_iou))
    selected_threshold = float(thresholds[best_threshold_index])
    test_metrics = collect_attention_metric_arrays(
        model,
        test_loader,
        device,
        [selected_threshold],
        minimum_pixels,
        compute_alignment=True,
    )
    if not np.array_equal(test_metrics["sample_id"], normal_predictions["sample_id"]):
        raise ValueError("Attention metric과 prediction의 test sample 순서가 다릅니다.")
    normal_correct = classification_metrics(
        normal_predictions["logits"],
        normal_predictions["labels"],
    )["correct_per_sample"].astype(np.float64)
    permuted_correct = np.full(normal_correct.shape, np.nan, dtype=np.float64)

    if model_name == "class_attention":
        permuted_predictions = collect_prediction_arrays(
            model,
            plain_test_loader,
            device,
            permute_attention=True,
        )
        permuted_correct = classification_metrics(
            permuted_predictions["logits"],
            permuted_predictions["labels"],
        )["correct_per_sample"].astype(np.float64)

    return {
        "sample_id": test_metrics["sample_id"],
        "pair_id": test_metrics["pair_id"],
        "overlap_level": test_metrics["overlap_level"],
        "average_precision": test_metrics["average_precision"],
        "iou": test_metrics["iou_by_threshold"][0],
        "cross_selectivity": test_metrics["cross_selectivity"],
        "normal_correct": normal_correct,
        "permuted_correct": permuted_correct,
        "validation_thresholds": thresholds,
        "validation_mean_iou": validation_mean_iou,
        "selected_iou_threshold": np.asarray(selected_threshold),
    }


def _save_cache(
    path: Path,
    arrays: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> None:
    """Run cache를 평가 fingerprint와 함께 압축 NPZ로 원자 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp.npz")
    payload = dict(arrays)
    payload["cache_fingerprint"] = np.asarray(evaluation_fingerprint(config))
    np.savez_compressed(temporary_path, **payload)
    temporary_path.replace(path)


def _load_cache(
    path: Path,
    config: ExperimentConfig,
    required_fields: set[str],
) -> dict[str, np.ndarray] | None:
    """NPZ가 현재 평가 계약과 호환될 때만 array를 반환한다.

    파일 손상, 누락 field, fingerprint 불일치는 모두 cache miss(None)로 처리한다.
    """
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not required_fields.issubset(archive.files):
                return None
            fingerprint = str(archive["cache_fingerprint"].item())
            if fingerprint != evaluation_fingerprint(config):
                return None
            return {
                field_name: archive[field_name]
                for field_name in archive.files
                if field_name != "cache_fingerprint"
            }
    except (OSError, ValueError, EOFError):
        return None


# -----------------------------------------------------------------------------
# Aggregate metric과 통계
# -----------------------------------------------------------------------------


def _write_aggregate_results(
    predictions_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
    attention_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
    config: ExperimentConfig,
) -> list[Path]:
    """현재 유효한 run cache에서 집계 CSV를 만들고, 전체 run이 모이면 최종 통계표를 추가한다."""
    model_metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for (model_name, seed), predictions in sorted(predictions_by_run.items()):
        _append_classification_rows(
            model_metric_rows,
            per_class_rows,
            model_name,
            seed,
            predictions,
        )

    model_metric_path = METRIC_LOG_DIR / "model_metrics.csv"
    per_class_metric_path = METRIC_LOG_DIR / "per_class_metrics.csv"
    model_cost_path = METRIC_LOG_DIR / "model_costs.csv"
    write_csv_rows(model_metric_path, model_metric_rows)
    write_csv_rows(per_class_metric_path, per_class_rows)
    write_csv_rows(model_cost_path, _model_cost_rows(config))
    generated_paths = [model_metric_path, per_class_metric_path, model_cost_path]

    attention_rows = [
        _attention_summary_row(model_name, seed, cache)
        for (model_name, seed), cache in sorted(attention_by_run.items())
    ]
    attention_metric_path = METRIC_LOG_DIR / "attention_metrics.csv"
    write_csv_rows(attention_metric_path, attention_rows)
    generated_paths.append(attention_metric_path)

    if not _all_run_caches_exist(predictions_by_run, attention_by_run, config):
        print("일부 run cache만 있어 전체 hierarchical 통계는 아직 생성하지 않습니다.")
        return generated_paths

    _validate_prediction_alignment(
        predictions_by_run,
        predictions_by_run[("lenet", config.project.training_seeds[0])],
    )
    _validate_attention_alignment(attention_by_run, predictions_by_run)
    _write_hierarchical_results(predictions_by_run, config)
    _write_per_class_recall_results(per_class_rows, config)
    _write_seed_effect_results(predictions_by_run, config)
    _write_training_stability(config)
    _write_class_pair_log(predictions_by_run, config)
    generated_paths.extend(
        (
            METRIC_LOG_DIR / "hierarchical_intervals.csv",
            METRIC_LOG_DIR / "per_class_recall.csv",
            METRIC_LOG_DIR / "seed_effects.csv",
            METRIC_LOG_DIR / "training_stability.csv",
            METRIC_LOG_DIR / "class_pair_accuracy_high.csv",
        )
    )
    return generated_paths


def _append_classification_rows(
    model_metric_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    model_name: str,
    seed: int,
    predictions: dict[str, np.ndarray],
) -> None:
    """한 run의 전체·overlap별 분류 metric과 class recall row를 추가한다."""
    for overlap_level in ("all", "low", "middle", "high"):
        mask = (
            np.ones(len(predictions["labels"]), dtype=bool)
            if overlap_level == "all"
            else predictions["overlap_level"] == overlap_level
        )
        metrics = classification_metrics(
            predictions["logits"][mask],
            predictions["labels"][mask],
        )
        model_metric_rows.append({
            "model": model_name,
            "seed": seed,
            "overlap_level": overlap_level,
            "sample_count": int(mask.sum()),
            "exact_match": metrics["exact_match"],
            "macro_f1": metrics["macro_f1"],
        })

        if overlap_level == "all":
            for class_index, (precision, recall) in enumerate(zip(
                metrics["per_class_precision"],
                metrics["per_class_recall"],
            )):
                per_class_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "class": class_index,
                    "precision": precision,
                    "recall": recall,
                })


def _write_hierarchical_results(
    predictions_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
    config: ExperimentConfig,
) -> None:
    """절대 정확도와 paired 효과의 seed·pair 2단계 bootstrap interval을 저장한다."""
    seeds = list(config.project.training_seeds)
    reference = predictions_by_run[("lenet", seeds[0])]
    correctness_by_run = _create_correctness_by_run(predictions_by_run)
    iterations = config.evaluation.bootstrap_iterations
    confidence_level = config.evaluation.confidence_level
    bootstrap_seed = config.project.data_seed
    effect_matrices: list[tuple[str, np.ndarray]] = []

    for model_name in config.model.model_names:
        for overlap_level in ("low", "middle", "high"):
            effect_matrices.append((
                f"{model_name}_{overlap_level}_accuracy",
                _level_matrix(
                    correctness_by_run,
                    model_name,
                    seeds,
                    reference,
                    overlap_level,
                ),
            ))
        effect_matrices.append((
            f"{model_name}_low_minus_high",
            _level_difference_matrix(
                correctness_by_run,
                model_name,
                seeds,
                reference,
            ),
        ))

    high_mask = reference["overlap_level"] == "high"
    for comparison_name, first_model, second_model in (
        ("class_attention_minus_lenet_high", "class_attention", "lenet"),
        (
            "class_attention_minus_shared_high",
            "class_attention",
            "shared_attention",
        ),
    ):
        matrix = np.stack([
            correctness_by_run[(first_model, seed)][high_mask]
            - correctness_by_run[(second_model, seed)][high_mask]
            for seed in seeds
        ])
        effect_matrices.append((comparison_name, matrix))

    did_matrix = []
    for seed in seeds:
        values, _ = difference_in_differences(
            correctness_by_run[("class_attention", seed)],
            correctness_by_run[("lenet", seed)],
            reference["pair_id"],
            reference["overlap_level"],
        )
        did_matrix.append(values)
    effect_matrices.append((
        "class_attention_vs_lenet_high_low_difference",
        np.stack(did_matrix),
    ))

    rows = []
    for comparison_index, (comparison, matrix) in enumerate(effect_matrices):
        estimate, lower, upper = hierarchical_bootstrap_interval(
            matrix,
            iterations,
            confidence_level,
            bootstrap_seed + comparison_index,
        )
        rows.append({
            "comparison": comparison,
            "estimate": estimate,
            "confidence_lower": lower,
            "confidence_upper": upper,
            "confidence_level": confidence_level,
            "bootstrap_iterations": iterations,
            "seed_count": matrix.shape[0],
            "pair_count": matrix.shape[1],
        })
    write_csv_rows(METRIC_LOG_DIR / "hierarchical_intervals.csv", rows)


def _level_matrix(
    correctness_by_run: dict[tuple[str, int], np.ndarray],
    model_name: str,
    seeds: list[int],
    reference: dict[str, np.ndarray],
    overlap_level: str,
) -> np.ndarray:
    """한 model·overlap의 correctness를 `[seed, pair]` 행렬로 만든다."""
    mask = reference["overlap_level"] == overlap_level
    return np.stack([
        correctness_by_run[(model_name, seed)][mask]
        for seed in seeds
    ])


def _level_difference_matrix(
    correctness_by_run: dict[tuple[str, int], np.ndarray],
    model_name: str,
    seeds: list[int],
    reference: dict[str, np.ndarray],
) -> np.ndarray:
    """한 model의 pair별 Low−High 효과를 `[seed, pair]` 행렬로 만든다."""
    rows = []
    for seed in seeds:
        differences, _ = paired_level_difference(
            correctness_by_run[(model_name, seed)],
            reference["pair_id"],
            reference["overlap_level"],
            "low",
            "high",
        )
        rows.append(differences)
    return np.stack(rows)


def _write_per_class_recall_results(
    per_class_rows: list[dict[str, Any]],
    config: ExperimentConfig,
) -> None:
    """숫자별 전체 test recall의 seed 평균·표준편차를 저장한다."""
    rows = []
    for class_index in range(CLASS_COUNT):
        model_values = {}
        for model_name in config.model.model_names:
            values = np.asarray([
                float(row["recall"])
                for row in per_class_rows
                if row["model"] == model_name and int(row["class"]) == class_index
            ])
            model_values[model_name] = values

        class_minus_lenet = (
            model_values["class_attention"] - model_values["lenet"]
        )
        row: dict[str, Any] = {"class": class_index}
        for model_name, values in model_values.items():
            row[f"{model_name}_mean"] = float(values.mean())
            row[f"{model_name}_standard_deviation"] = sample_deviation(values)
        row["class_attention_minus_lenet_mean"] = float(class_minus_lenet.mean())
        row["class_attention_minus_lenet_standard_deviation"] = (
            sample_deviation(class_minus_lenet)
        )
        row["seed_count"] = len(class_minus_lenet)
        rows.append(row)
    write_csv_rows(METRIC_LOG_DIR / "per_class_recall.csv", rows)


def _write_seed_effect_results(
    predictions_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
    config: ExperimentConfig,
) -> None:
    """Appendix용 seed별 paired test-pair bootstrap 결과를 저장한다."""
    seeds = list(config.project.training_seeds)
    reference = predictions_by_run[("lenet", seeds[0])]
    rows = create_seed_effect_rows(
        _create_correctness_by_run(predictions_by_run),
        reference["pair_id"],
        reference["overlap_level"],
        seeds,
        config.evaluation.bootstrap_iterations,
        config.evaluation.confidence_level,
        config.project.data_seed,
    )
    write_csv_rows(METRIC_LOG_DIR / "seed_effects.csv", rows)


def _write_training_stability(config: ExperimentConfig) -> None:
    """완료 checkpoint와 epoch history에서 early-stopping 진단표를 만든다."""
    rows = []
    maximum_epochs = config.train.maximum_epochs
    for model_name in config.model.model_names:
        for seed in config.project.training_seeds:
            history_path = TRAINING_LOG_DIR / f"{model_name}_seed_{seed}.csv"
            checkpoint_path = CHECKPOINT_DIR / f"{model_name}_seed_{seed}.pt"
            with history_path.open(newline="", encoding="utf-8") as history_file:
                history_rows = list(csv.DictReader(history_file))
            if not history_rows:
                raise ValueError(f"Training history가 비어 있습니다: {history_path}")
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint_is_valid = (
                checkpoint.get("config_fingerprint") == config_fingerprint(config)
                and checkpoint.get("training_complete") is True
            )
            if not checkpoint_is_valid:
                raise ValueError(f"완료되지 않은 checkpoint입니다: {checkpoint_path}")
            final_row = history_rows[-1]
            epochs_run = int(checkpoint["epochs_run"])
            best_epoch = int(checkpoint["best_epoch"])
            rows.append({
                "model": model_name,
                "seed": seed,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "best_validation_exact_match": float(
                    checkpoint["validation_exact_match"]
                ),
                "final_validation_exact_match": float(
                    final_row["validation_exact_match"]
                ),
                "reached_maximum_epochs": epochs_run == maximum_epochs,
                "best_at_final_epoch": best_epoch == epochs_run,
                "training_complete": True,
            })
    write_csv_rows(METRIC_LOG_DIR / "training_stability.csv", rows)


def _write_class_pair_log(
    predictions_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
    config: ExperimentConfig,
) -> None:
    """High-overlap 45개 class pair의 seed 평균 accuracy log를 저장한다."""
    seeds = list(config.project.training_seeds)
    reference = predictions_by_run[("lenet", seeds[0])]
    high_mask = reference["overlap_level"] == "high"
    correctness_by_run = _create_correctness_by_run(predictions_by_run)
    rows = []
    for model_name in config.model.model_names:
        mean_correctness = np.mean([
            correctness_by_run[(model_name, seed)]
            for seed in seeds
        ], axis=0)
        matrix = class_pair_accuracy(
            mean_correctness[high_mask],
            reference["label_first"][high_mask],
            reference["label_second"][high_mask],
        )
        for first_class in range(CLASS_COUNT):
            for second_class in range(first_class + 1, CLASS_COUNT):
                rows.append({
                    "model": model_name,
                    "label_first": first_class,
                    "label_second": second_class,
                    "accuracy": matrix[first_class, second_class],
                })
    write_csv_rows(METRIC_LOG_DIR / "class_pair_accuracy_high.csv", rows)


# -----------------------------------------------------------------------------
# 공통 helper
# -----------------------------------------------------------------------------


def _attention_summary_row(
    model_name: str,
    seed: int,
    cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Sample attention cache를 한 model·seed 집계 row로 변환한다."""
    valid_selectivity = np.isfinite(cache["cross_selectivity"])
    permutation_drop: float | str = ""
    permuted_exact_match: float | str = ""
    if np.isfinite(cache["permuted_correct"]).any():
        permuted_exact_match = finite_mean(cache["permuted_correct"])
        permutation_drop = (
            finite_mean(cache["normal_correct"]) - float(permuted_exact_match)
        )
    return {
        "model": model_name,
        "seed": seed,
        "selected_iou_threshold": float(cache["selected_iou_threshold"].item()),
        "test_average_precision": finite_mean(cache["average_precision"]),
        "test_iou": finite_mean(cache["iou"]),
        "test_cross_selectivity": finite_mean(cache["cross_selectivity"]),
        "selectivity_valid_fraction": float(valid_selectivity.mean()),
        "normal_exact_match": finite_mean(cache["normal_correct"]),
        "permuted_exact_match": permuted_exact_match,
        "permutation_accuracy_drop": permutation_drop,
    }


def _model_cost_rows(config: ExperimentConfig) -> list[dict[str, Any]]:
    """세 모델의 trainable parameter 수와 forward MAC row를 만든다."""
    canvas_size = config.dataset.canvas_size
    rows = []
    for model_name in config.model.model_names:
        model = create_model(model_name, config)
        rows.append({
            "model": model_name,
            "parameters": count_parameters(model),
            "multiply_accumulate_operations": estimate_multiply_accumulate_operations(
                model,
                (1, 1, canvas_size, canvas_size),
            ),
        })
    return rows


def _load_all_prediction_caches(
    config: ExperimentConfig,
    require_all: bool,
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Config에 속한 유효 prediction cache를 모두 읽는다."""
    caches = {}
    for model_name in config.model.model_names:
        for seed in config.project.training_seeds:
            path = _prediction_cache_path(model_name, seed)
            cache = _load_cache(path, config, PREDICTION_CACHE_FIELDS)
            if cache is None and require_all:
                raise FileNotFoundError(
                    f"유효한 prediction cache가 없습니다: {path}. evaluate를 실행하세요."
                )
            if cache is not None:
                caches[(model_name, seed)] = cache
    return caches


def _load_all_attention_caches(
    config: ExperimentConfig,
    require_all: bool,
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """두 attention 모델의 유효 sample metric cache를 모두 읽는다."""
    caches = {}
    for model_name in ("shared_attention", "class_attention"):
        for seed in config.project.training_seeds:
            path = _attention_cache_path(model_name, seed)
            cache = _load_cache(path, config, ATTENTION_CACHE_FIELDS)
            if cache is None and require_all:
                raise FileNotFoundError(
                    f"유효한 attention cache가 없습니다: {path}. evaluate를 실행하세요."
                )
            if cache is not None:
                caches[(model_name, seed)] = cache
    return caches


def _all_run_caches_exist(
    predictions_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
    attention_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
    config: ExperimentConfig,
) -> bool:
    """최종 집계에 필요한 prediction·attention cache가 모두 있는지 확인한다."""
    seeds = list(config.project.training_seeds)
    expected_predictions = {
        (model_name, seed)
        for model_name in config.model.model_names
        for seed in seeds
    }
    expected_attention = {
        (model_name, seed)
        for model_name in ("shared_attention", "class_attention")
        for seed in seeds
    }
    return (
        expected_predictions.issubset(predictions_by_run)
        and expected_attention.issubset(attention_by_run)
    )


def _create_correctness_by_run(
    predictions_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
) -> dict[tuple[str, int], np.ndarray]:
    """Prediction logit을 sample별 exact-match Float64 array로 바꾼다."""
    return {
        run_key: classification_metrics(
            predictions["logits"],
            predictions["labels"],
        )["correct_per_sample"].astype(np.float64)
        for run_key, predictions in predictions_by_run.items()
    }


def _validate_prediction_alignment(
    predictions_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
    reference: dict[str, np.ndarray],
) -> None:
    """모든 prediction cache가 같은 test sample과 순서를 사용하는지 검사한다."""
    fields = ("sample_id", "pair_id", "labels", "overlap_level")
    for run_key, predictions in predictions_by_run.items():
        for field_name in fields:
            if not np.array_equal(predictions[field_name], reference[field_name]):
                raise ValueError(
                    f"Prediction sample 정렬이 다릅니다: run={run_key}, field={field_name}"
                )


def _validate_attention_alignment(
    attention_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
    predictions_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
) -> None:
    """Attention cache가 같은 run의 prediction sample 순서를 따르는지 검사한다."""
    for run_key, attention_cache in attention_by_run.items():
        prediction_cache = predictions_by_run[run_key]
        for field_name in ("sample_id", "pair_id", "overlap_level"):
            if not np.array_equal(
                attention_cache[field_name],
                prediction_cache[field_name],
            ):
                raise ValueError(
                    f"Attention sample 정렬이 다릅니다: run={run_key}, "
                    f"field={field_name}"
                )


def _remove_evaluation_caches(model_names: list[str], seeds: list[int]) -> None:
    """선택한 run의 prediction·attention cache만 삭제한다 (checkpoint는 유지)."""
    for model_name in model_names:
        for seed in seeds:
            for cache_path in (
                _prediction_cache_path(model_name, seed),
                _attention_cache_path(model_name, seed),
            ):
                cache_path.unlink(missing_ok=True)
                cache_path.with_name(
                    cache_path.name + ".tmp.npz"
                ).unlink(missing_ok=True)


def _create_attention_loaders(
    config: ExperimentConfig,
    batch_size: int,
    workers: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Mask를 포함한 validation·test DataLoader를 고정 순서로 만든다."""
    loaders = []
    for split in ("validation", "test"):
        loaders.append(torch.utils.data.DataLoader(
            ControlledOverlapMnistDataset(split, config, include_masks=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        ))
    return loaders[0], loaders[1]


def _attention_thresholds(evaluation_config: EvaluationConfig) -> np.ndarray:
    """start·stop·step 설정을 오름차순 IoU threshold array로 변환한다."""
    thresholds = evaluation_config.attention_iou_thresholds
    return np.round(
        np.arange(thresholds.start, thresholds.stop + thresholds.step * 0.5, thresholds.step),
        decimals=10,
    )


def _prediction_cache_path(model_name: str, seed: int) -> Path:
    """한 run의 prediction cache NPZ 경로."""
    return PREDICTION_LOG_DIR / f"{model_name}_seed_{seed}_test.npz"


def _attention_cache_path(model_name: str, seed: int) -> Path:
    """한 run의 attention metric cache NPZ 경로."""
    return ATTENTION_LOG_DIR / f"{model_name}_seed_{seed}_test.npz"
