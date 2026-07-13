"""학습 checkpoint를 평가하고 prediction 및 metric log를 생성한다.

입력:
    Config 경로, model/seed 선택, 실행 device

출력:
    Prediction NPZ와 분류·attention·bootstrap CSV 경로

연결:
    CLI와 report generator가 evaluation package의 공개 함수로 사용한다.
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..configuration import (
    CHECKPOINT_DIR,
    DEFAULT_CONFIG_PATH,
    METRIC_LOG_DIR,
    PREDICTION_LOG_DIR,
    create_output_directories,
    load_config,
)
from ..data import ControlledOverlapMnistDataset
from ..models import create_model
from ..training import load_checkpoint
from .analysis import (
    collect_prediction_arrays,
    count_parameters,
    difference_in_differences,
    evaluate_attention_loader,
    estimate_multiply_accumulate_operations,
    finite_mean,
    paired_bootstrap_interval,
    paired_level_difference,
    select_best_iou_threshold,
)
from .metrics import class_pair_accuracy, classification_metrics


# -----------------------------------------------------------------------------
# 전체 평가 실행
# -----------------------------------------------------------------------------


def evaluate_models(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_name: str | None = None,
    seed: int | None = None,
    device_name: str = "cpu",
) -> list[Path]:
    """선택한 checkpoint를 test set에서 평가하고 분석 log를 저장한다.

    입력:
        Config 경로, 선택 model/seed, `cpu` 또는 `cuda`

    처리:
        Prediction, 분류 지표, attention 정렬, 비용 및 전체 비교를 계산한다.

    출력:
        생성된 prediction과 metric file 경로 목록
    """
    config = load_config(config_path)
    create_output_directories()
    model_names = (
        list(config["model"]["model_names"])
        if model_name is None
        else [model_name]
    )
    seeds = (
        [int(configured_seed) for configured_seed in config["project"]["training_seeds"]]
        if seed is None
        else [int(seed)]
    )
    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 현재 환경에서 사용할 수 없습니다.")

    evaluation_config = config["evaluation"]
    batch_size = int(evaluation_config["batch_size"])
    workers = int(config["train"]["data_loader_workers"])
    test_loader = torch.utils.data.DataLoader(
        ControlledOverlapMnistDataset("test", config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    validation_attention_loader = torch.utils.data.DataLoader(
        ControlledOverlapMnistDataset("validation", config, include_masks=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    test_attention_loader = torch.utils.data.DataLoader(
        ControlledOverlapMnistDataset("test", config, include_masks=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    model_metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    attention_metric_rows: list[dict[str, Any]] = []
    model_cost_rows: list[dict[str, Any]] = []
    predictions_by_run: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    thresholds = _attention_thresholds(evaluation_config)
    generated_paths: list[Path] = []

    for model_name in model_names:
        for seed in seeds:
            checkpoint_path = CHECKPOINT_DIR / f"{model_name}_seed_{seed}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint가 없습니다: {checkpoint_path}")
            model = create_model(model_name, config)
            load_checkpoint(model, checkpoint_path, device, config)
            predictions = collect_prediction_arrays(model, test_loader, device)
            predictions_by_run[(model_name, int(seed))] = predictions
            prediction_path = PREDICTION_LOG_DIR / f"{model_name}_seed_{seed}_test.npz"
            np.savez_compressed(prediction_path, **predictions)
            generated_paths.append(prediction_path)
            _append_classification_rows(
                model_metric_rows,
                per_class_rows,
                model_name,
                int(seed),
                predictions,
            )

            if model_name in ("shared_attention", "class_attention"):
                attention_metric_rows.append(_evaluate_attention_run(
                    model,
                    model_name,
                    int(seed),
                    validation_attention_loader,
                    test_attention_loader,
                    test_loader,
                    thresholds,
                    evaluation_config,
                    device,
                    predictions,
                ))
            print(f"평가 완료: model={model_name}, seed={seed}, file={prediction_path}")

        cost_model = create_model(model_name, config)
        canvas_size = int(config["dataset"]["canvas_size"])
        model_cost_rows.append({
            "model": model_name,
            "parameters": count_parameters(cost_model),
            "multiply_accumulate_operations": estimate_multiply_accumulate_operations(
                cost_model, (1, 1, canvas_size, canvas_size)
            ),
        })

    metric_paths = (
        METRIC_LOG_DIR / "model_metrics.csv",
        METRIC_LOG_DIR / "per_class_metrics.csv",
        METRIC_LOG_DIR / "model_costs.csv",
    )
    _write_csv(metric_paths[0], model_metric_rows)
    _write_csv(metric_paths[1], per_class_rows)
    _write_csv(metric_paths[2], model_cost_rows)
    generated_paths.extend(metric_paths)

    if attention_metric_rows:
        attention_path = METRIC_LOG_DIR / "attention_metrics.csv"
        _write_csv(attention_path, attention_metric_rows)
        generated_paths.append(attention_path)

    evaluated_all_models = set(config["model"]["model_names"]).issubset(model_names)
    evaluated_all_seeds = set(config["project"]["training_seeds"]).issubset(seeds)

    if evaluated_all_models and evaluated_all_seeds:
        _write_cross_model_results(predictions_by_run, config)
        generated_paths.extend(
            (
                METRIC_LOG_DIR / "bootstrap_intervals.csv",
                METRIC_LOG_DIR / "class_pair_accuracy_high.csv",
            )
        )

    return generated_paths


# -----------------------------------------------------------------------------
# 실행별 metric 구성
# -----------------------------------------------------------------------------


def _append_classification_rows(
    model_metric_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    model_name: str,
    seed: int,
    predictions: dict[str, np.ndarray],
) -> None:
    """한 model/seed prediction의 전체 및 overlap별 metric row를 추가한다.

    입력:
        수정할 두 row 목록, model 이름, seed, prediction dictionary

    처리:
        네 평가 범위의 분류 지표와 전체 범위의 class별 precision/recall을 계산한다.

    출력:
        반환값은 없으며 입력 row 목록 두 개가 제자리에서 확장된다.
    """
    for overlap_level in ("all", "low", "middle", "high"):
        if overlap_level == "all":
            mask = np.ones(len(predictions["labels"]), dtype=bool)
        else:
            mask = predictions["overlap_level"] == overlap_level

        metrics = classification_metrics(
            predictions["logits"][mask],
            predictions["labels"][mask],
        )
        model_metric_rows.append(
            {
                "model": model_name,
                "seed": seed,
                "overlap_level": overlap_level,
                "sample_count": int(mask.sum()),
                "exact_match": metrics["exact_match"],
                "macro_f1": metrics["macro_f1"],
            }
        )

        if overlap_level == "all":
            precision_recall_pairs = zip(
                metrics["per_class_precision"],
                metrics["per_class_recall"],
            )
            for class_index, (precision, recall) in enumerate(precision_recall_pairs):
                per_class_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "class": class_index,
                        "precision": precision,
                        "recall": recall,
                    }
                )


def _evaluate_attention_run(
    model: torch.nn.Module,
    model_name: str,
    seed: int,
    validation_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    plain_test_loader: torch.utils.data.DataLoader,
    thresholds: np.ndarray,
    evaluation_config: dict[str, Any],
    device: torch.device,
    normal_predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """한 attention checkpoint의 validation threshold와 test metric을 계산한다.

    입력:
        Model metadata, validation/test loader, threshold, config, device, 정상 prediction

    처리:
        Validation에서 IoU threshold를 선택하고 test 정렬 및 permutation 성능을 구한다.

    출력:
        `attention_metrics.csv`에 기록할 한 row dictionary
    """
    validation_results = evaluate_attention_loader(
        model,
        validation_loader,
        device,
        thresholds,
        int(evaluation_config["minimum_exclusive_pixels"]),
        compute_alignment=False,
    )
    selected_threshold = select_best_iou_threshold(validation_results["iou_counts"])
    test_results = evaluate_attention_loader(
        model,
        test_loader,
        device,
        [selected_threshold],
        int(evaluation_config["minimum_exclusive_pixels"]),
    )
    iou_sum, iou_count = test_results["iou_counts"][selected_threshold]
    row: dict[str, Any] = {
        "model": model_name,
        "seed": seed,
        "selected_iou_threshold": selected_threshold,
        "test_average_precision": finite_mean(test_results["average_precision"]),
        "test_iou": iou_sum / max(iou_count, 1),
        "test_cross_selectivity": finite_mean(test_results["cross_selectivity"]),
        "selectivity_valid_fraction": test_results["selectivity_valid_fraction"],
        "permuted_exact_match": "",
        "permutation_accuracy_drop": "",
    }
    if model_name == "class_attention":
        permuted_predictions = collect_prediction_arrays(
            model, plain_test_loader, device, permute_attention=True
        )
        normal_accuracy = classification_metrics(
            normal_predictions["logits"], normal_predictions["labels"]
        )["exact_match"]
        permuted_accuracy = classification_metrics(
            permuted_predictions["logits"], permuted_predictions["labels"]
        )["exact_match"]
        row["permuted_exact_match"] = permuted_accuracy
        row["permutation_accuracy_drop"] = normal_accuracy - permuted_accuracy
    return row


# -----------------------------------------------------------------------------
# 모델 간 paired 비교와 CSV 저장
# -----------------------------------------------------------------------------


def _write_cross_model_results(
    predictions_by_run: dict[tuple[str, int], dict[str, np.ndarray]],
    config: dict[str, Any],
) -> None:
    """세 모델과 전체 seed가 있을 때 primary 비교와 class-pair 결과를 저장한다.

    입력:
        `(model, seed)`별 prediction dictionary와 전체 config

    처리:
        Seed correctness 평균, paired bootstrap, difference-in-differences를 계산한다.

    출력:
        반환값은 없으며 bootstrap 및 class-pair CSV 두 개를 생성한다.
    """
    seeds = [int(seed) for seed in config["project"]["training_seeds"]]
    correctness_by_model = {}
    reference_predictions = predictions_by_run[("lenet", seeds[0])]
    for model_name in config["model"]["model_names"]:
        seed_correctness = []
        for seed in seeds:
            predictions = predictions_by_run[(model_name, seed)]
            metrics = classification_metrics(predictions["logits"], predictions["labels"])
            seed_correctness.append(metrics["correct_per_sample"].astype(np.float64))
        correctness_by_model[model_name] = np.mean(seed_correctness, axis=0)

    high_mask = reference_predictions["overlap_level"] == "high"
    baseline_differences, baseline_pair_ids = paired_level_difference(
        correctness=correctness_by_model["lenet"],
        pair_ids=reference_predictions["pair_id"],
        overlap_levels=reference_predictions["overlap_level"],
        first_level="low",
        second_level="high",
    )
    estimate, lower, upper = paired_bootstrap_interval(
        baseline_differences,
        baseline_pair_ids,
        int(config["evaluation"]["bootstrap_iterations"]),
        0.95,
        int(config["project"]["data_seed"]),
    )
    bootstrap_rows = [
        _bootstrap_row("lenet_low_minus_high", estimate, lower, upper)
    ]

    comparisons = (
        ("class_attention_minus_lenet_high", "class_attention", "lenet"),
        ("class_attention_minus_shared_high", "class_attention", "shared_attention"),
    )
    for comparison_name, first_model, second_model in comparisons:
        differences = (
            correctness_by_model[first_model][high_mask]
            - correctness_by_model[second_model][high_mask]
        )
        estimate, lower, upper = paired_bootstrap_interval(
            differences,
            reference_predictions["pair_id"][high_mask],
            int(config["evaluation"]["bootstrap_iterations"]),
            0.95,
            int(config["project"]["data_seed"]),
        )
        bootstrap_rows.append(_bootstrap_row(comparison_name, estimate, lower, upper))

    did_values, did_pair_ids = difference_in_differences(
        correctness_by_model["class_attention"],
        correctness_by_model["lenet"],
        reference_predictions["pair_id"],
        reference_predictions["overlap_level"],
    )
    estimate, lower, upper = paired_bootstrap_interval(
        did_values,
        did_pair_ids,
        int(config["evaluation"]["bootstrap_iterations"]),
        0.95,
        int(config["project"]["data_seed"]),
    )
    bootstrap_rows.append(_bootstrap_row(
        "class_attention_vs_lenet_high_low_difference", estimate, lower, upper
    ))
    _write_csv(METRIC_LOG_DIR / "bootstrap_intervals.csv", bootstrap_rows)

    class_pair_rows = []
    for model_name, correctness in correctness_by_model.items():
        matrix = class_pair_accuracy(
            correctness[high_mask],
            reference_predictions["label_first"][high_mask],
            reference_predictions["label_second"][high_mask],
        )
        for first_class in range(10):
            for second_class in range(first_class + 1, 10):
                class_pair_rows.append({
                    "model": model_name,
                    "label_first": first_class,
                    "label_second": second_class,
                    "accuracy": matrix[first_class, second_class],
                })
    _write_csv(METRIC_LOG_DIR / "class_pair_accuracy_high.csv", class_pair_rows)


# -----------------------------------------------------------------------------
# Config 변환과 출력 helper
# -----------------------------------------------------------------------------


def _attention_thresholds(evaluation_config: dict[str, Any]) -> np.ndarray:
    """Config의 start/stop/step을 validation IoU threshold array로 변환한다.

    입력:
        Config의 evaluation section

    처리:
        Floating-point 끝값을 포함하도록 NumPy range를 생성한다.

    출력:
        오름차순 threshold array
    """
    threshold_config = evaluation_config["attention_iou_thresholds"]
    return np.arange(
        float(threshold_config["start"]),
        float(threshold_config["stop"]) + float(threshold_config["step"]) / 2.0,
        float(threshold_config["step"]),
    )


def _bootstrap_row(name: str, estimate: float, lower: float, upper: float) -> dict[str, Any]:
    """Bootstrap 결과를 CSV에 기록할 dictionary로 묶는다.

    입력:
        비교 이름, 추정값, confidence 하한과 상한

    처리:
        고정 column 이름에 값을 대응시킨다.

    출력:
        한 개 bootstrap CSV row
    """
    return {
        "comparison": name,
        "estimate": estimate,
        "confidence_lower": lower,
        "confidence_upper": upper,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Dictionary row 목록을 UTF-8 CSV로 저장한다.

    입력:
        출력 경로와 동일 key를 가진 row 목록

    처리:
        첫 row key를 header로 사용해 전체 row를 기록한다.

    출력:
        반환값은 없으며 row가 있을 때 CSV가 생성된다.
    """
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
