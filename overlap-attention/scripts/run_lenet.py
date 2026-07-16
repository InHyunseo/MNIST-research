#!/usr/bin/env python
"""LeNet 단독 실험: 데이터 준비 → 학습 → 평가 → 발표용 수치 로그.

발표(5분)와 구술 평가를 위해 `ModernizedLeNet` 한 모델만 config의 전체 seed로
학습하고, LeNet에 필요한 수치만 직접 계산해 stdout과 `results/lenet_metrics.json`에
남긴다. Attention 모델을 함께 요구하는 기존 3-model 집계(`evaluate_models`의
hierarchical 통계)는 건드리지 않고 우회한다.

실행:
    .venv/bin/python scripts/run_lenet.py [--config PATH] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mnist_overlap.analysis import (  # noqa: E402
    collect_prediction_arrays,
    hierarchical_bootstrap_interval,
    paired_bootstrap_interval,
    paired_level_difference,
)
from mnist_overlap.config import (  # noqa: E402
    CHECKPOINT_DIR,
    CLASS_COUNT,
    DEFAULT_CONFIG_PATH,
    SUMMARY_PATH,
    create_output_directories,
    load_config,
)
from mnist_overlap.data import (  # noqa: E402
    DATA_FINGERPRINT_PATH,
    ControlledOverlapMnistDataset,
    prepare_data,
)
from mnist_overlap.config import data_config_fingerprint  # noqa: E402
from mnist_overlap.metrics import (  # noqa: E402
    class_pair_accuracy,
    classification_metrics,
    exact_match_per_sample,
    sample_deviation,
)
from mnist_overlap.models import create_model  # noqa: E402
from mnist_overlap.training import (  # noqa: E402
    load_checkpoint,
    select_device,
    train_models,
)

import torch  # noqa: E402

MODEL_NAME = "lenet"
OVERLAP_LEVELS = ("low", "middle", "high")
RESULTS_JSON_PATH = PROJECT_ROOT / "results" / "lenet_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="LeNet 단독 실험을 실행합니다.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="기존 checkpoint를 재사용하고 학습 단계를 건너뜁니다.",
    )
    args = parser.parse_args()

    # 파일·파이프로 리다이렉트해도 진행 로그가 실시간으로 보이게 한다.
    sys.stdout.reconfigure(line_buffering=True)

    config = load_config(args.config)
    create_output_directories()
    device = select_device(args.device)
    seeds = list(config.project.training_seeds)

    print("=" * 70)
    print(f"LeNet 단독 실험 — seeds={seeds}, device={args.device}")
    print("=" * 70)

    print("\n[1/3] 데이터 준비 (manifest fingerprint 검사 후 필요 시 재생성)")
    saved_fingerprint = (
        DATA_FINGERPRINT_PATH.read_text(encoding="utf-8").strip()
        if DATA_FINGERPRINT_PATH.exists()
        else None
    )
    needs_regen = saved_fingerprint != data_config_fingerprint(config)
    if needs_regen:
        print("  현재 config와 manifest fingerprint 불일치 → manifest를 재생성합니다.")
    prepare_data(config, overwrite=needs_regen)

    if not args.skip_training:
        print(f"\n[2/3] 학습 — LeNet × {len(seeds)} seeds")
        train_models(
            config_path=args.config,
            model_name=MODEL_NAME,
            device_name=args.device,
            overwrite=False,
        )
    else:
        print("\n[2/3] 학습 생략 (--skip-training)")

    print("\n[3/3] 평가 — test prediction 수집 및 수치 계산")
    predictions_by_seed = _collect_predictions(config, seeds, device)

    metrics = _compute_metrics(config, predictions_by_seed, seeds)
    _print_report(metrics)

    RESULTS_JSON_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n수치를 저장했습니다: {RESULTS_JSON_PATH}")
    print(f"(발표 원문은 {SUMMARY_PATH.parent}/PRESENTATION.md 를 참조)")


def _collect_predictions(
    config,
    seeds: list[int],
    device: torch.device,
) -> dict[int, dict[str, np.ndarray]]:
    """각 seed의 best checkpoint로 test set logit·label·metadata를 수집한다."""
    test_dataset = ControlledOverlapMnistDataset("test", config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.train.data_loader_workers,
    )
    predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        checkpoint_path = CHECKPOINT_DIR / f"{MODEL_NAME}_seed_{seed}.pt"
        model = create_model(MODEL_NAME, config)
        load_checkpoint(model, checkpoint_path, device, config)
        predictions_by_seed[seed] = collect_prediction_arrays(
            model, test_loader, device
        )
        print(f"  seed={seed} prediction 수집 완료")
    return predictions_by_seed


def _compute_metrics(
    config,
    predictions_by_seed: dict[int, dict[str, np.ndarray]],
    seeds: list[int],
) -> dict:
    """LeNet 발표에 필요한 모든 수치를 하나의 dictionary로 계산한다."""
    reference = predictions_by_seed[seeds[0]]
    iterations = config.evaluation.bootstrap_iterations
    confidence = config.evaluation.confidence_level
    bootstrap_seed = config.project.data_seed

    # --- overlap별 분류 성능: seed 평균 ± 표본 표준편차 ---
    performance: dict[str, dict[str, float]] = {}
    per_seed_scores: dict[str, dict[str, list[float]]] = {}
    for level in ("all", *OVERLAP_LEVELS):
        exacts, f1s = [], []
        for seed in seeds:
            preds = predictions_by_seed[seed]
            mask = (
                np.ones(len(preds["labels"]), dtype=bool)
                if level == "all"
                else preds["overlap_level"] == level
            )
            m = classification_metrics(preds["logits"][mask], preds["labels"][mask])
            exacts.append(m["exact_match"])
            f1s.append(m["macro_f1"])
        exacts_arr = np.asarray(exacts)
        f1s_arr = np.asarray(f1s)
        performance[level] = {
            "exact_match_mean": float(exacts_arr.mean()),
            "exact_match_std": sample_deviation(exacts_arr),
            "macro_f1_mean": float(f1s_arr.mean()),
            "macro_f1_std": sample_deviation(f1s_arr),
        }
        per_seed_scores[level] = {"exact_match": exacts, "macro_f1": f1s}

    correctness_by_seed = {
        seed: _correct_per_sample(predictions_by_seed[seed]) for seed in seeds
    }

    # --- Headline: Low − High, seed+pair 2단계 hierarchical bootstrap ---
    low_high_matrix = np.stack([
        paired_level_difference(
            correctness_by_seed[seed],
            reference["pair_id"],
            reference["overlap_level"],
            "low",
            "high",
        )[0]
        for seed in seeds
    ])
    lh_estimate, lh_lower, lh_upper = hierarchical_bootstrap_interval(
        low_high_matrix, iterations, confidence, bootstrap_seed
    )

    # --- Appendix: seed별 Low − High (부호 안정성) ---
    per_seed_low_high = []
    for index, seed in enumerate(seeds):
        differences, pair_ids = paired_level_difference(
            correctness_by_seed[seed],
            reference["pair_id"],
            reference["overlap_level"],
            "low",
            "high",
        )
        estimate, lower, upper = paired_bootstrap_interval(
            differences, pair_ids, iterations, confidence, bootstrap_seed + index
        )
        per_seed_low_high.append({
            "seed": seed,
            "estimate": estimate,
            "confidence_lower": lower,
            "confidence_upper": upper,
        })

    # --- Failure analysis (High overlap) ---
    failure = _failure_analysis(predictions_by_seed, seeds)

    return {
        "model": MODEL_NAME,
        "seeds": seeds,
        "bootstrap_iterations": iterations,
        "confidence_level": confidence,
        "performance": performance,
        "per_seed_scores": per_seed_scores,
        "headline_low_minus_high": {
            "estimate": lh_estimate,
            "confidence_lower": lh_lower,
            "confidence_upper": lh_upper,
            "seed_count": int(low_high_matrix.shape[0]),
            "pair_count": int(low_high_matrix.shape[1]),
        },
        "per_seed_low_minus_high": per_seed_low_high,
        "failure_analysis_high": failure,
    }


def _failure_analysis(
    predictions_by_seed: dict[int, dict[str, np.ndarray]],
    seeds: list[int],
) -> dict:
    """High overlap에서 가장 어렵/쉬운 class pair와 전체 최저 recall class를 찾는다.

    class-pair 정확도는 각 seed의 High-overlap sample correctness를 pair별로 평균한
    뒤 seed 평균을 취한다. recall은 전체 test에서 seed 평균을 취한다.
    """
    reference = predictions_by_seed[seeds[0]]
    high_mask = reference["overlap_level"] == "high"
    label_first_high = reference["label_first"][high_mask]
    label_second_high = reference["label_second"][high_mask]

    pair_matrices = []
    recall_by_seed = []
    for seed in seeds:
        preds = predictions_by_seed[seed]
        correct = _correct_per_sample(preds)
        pair_matrices.append(
            class_pair_accuracy(
                correct[high_mask],
                label_first_high,
                label_second_high,
                CLASS_COUNT,
            )
        )
        recall_by_seed.append(
            classification_metrics(preds["logits"], preds["labels"])[
                "per_class_recall"
            ]
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_pair_matrix = np.nanmean(np.stack(pair_matrices), axis=0)
    mean_recall = np.stack(recall_by_seed).mean(axis=0)

    pair_entries = []
    for first in range(CLASS_COUNT):
        for second in range(first + 1, CLASS_COUNT):
            value = mean_pair_matrix[first, second]
            if np.isfinite(value):
                pair_entries.append((first, second, float(value)))
    hardest = min(pair_entries, key=lambda item: item[2])
    easiest = max(pair_entries, key=lambda item: item[2])
    weakest_class = int(np.argmin(mean_recall))

    return {
        "hardest_pair": {
            "first": hardest[0],
            "second": hardest[1],
            "accuracy": hardest[2],
        },
        "easiest_pair": {
            "first": easiest[0],
            "second": easiest[1],
            "accuracy": easiest[2],
        },
        "lowest_recall_class_overall": {
            "class": weakest_class,
            "recall": float(mean_recall[weakest_class]),
        },
    }


def _correct_per_sample(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Top-2 예측 집합이 정답과 정확히 일치하는지 sample별 float 배열로 반환한다."""
    logits = torch.from_numpy(predictions["logits"])
    labels = torch.from_numpy(predictions["labels"])
    return exact_match_per_sample(logits, labels).numpy().astype(np.float64)


def _print_report(metrics: dict) -> None:
    """계산한 수치를 발표 슬롯 순서대로 stdout에 정리한다."""
    percent = metrics["confidence_level"] * 100.0
    print("\n" + "=" * 70)
    print("LeNet 결과 요약")
    print("=" * 70)

    print("\n[Top-2 exact-match accuracy (%)] seed 평균 ± 표본 std")
    perf = metrics["performance"]
    header = f"{'':8} {'Overall':>16} {'Low':>16} {'Middle':>16} {'High':>16}"
    print(header)
    exact_cells = " ".join(
        f"{perf[level]['exact_match_mean'] * 100:6.2f} ± "
        f"{perf[level]['exact_match_std'] * 100:5.2f}"
        for level in ("all", "low", "middle", "high")
    )
    print(f"{'exact':8} {exact_cells}")
    f1_cells = " ".join(
        f"{perf[level]['macro_f1_mean']:6.4f} ± "
        f"{perf[level]['macro_f1_std']:6.4f}"
        for level in ("all", "low", "middle", "high")
    )
    print(f"\n[Macro-F1] seed 평균 ± 표본 std")
    print(header)
    print(f"{'f1':8} {f1_cells}")

    head = metrics["headline_low_minus_high"]
    print(
        f"\n[Headline] Low − High = {head['estimate'] * 100:.2f}%p, "
        f"{percent:.0f}% CI "
        f"[{head['confidence_lower'] * 100:.2f}, {head['confidence_upper'] * 100:.2f}] "
        f"(seed+pair hierarchical bootstrap, {metrics['bootstrap_iterations']:,}회)"
    )

    signs = [row["estimate"] for row in metrics["per_seed_low_minus_high"]]
    positive = sum(1 for value in signs if value > 0)
    print(
        f"  seed별 Low − High: {positive}/{len(signs)}개 seed가 양수 "
        f"(범위 {min(signs) * 100:.2f} ~ {max(signs) * 100:.2f}%p)"
    )

    fail = metrics["failure_analysis_high"]
    hp = fail["hardest_pair"]
    ep = fail["easiest_pair"]
    lr = fail["lowest_recall_class_overall"]
    print("\n[Failure analysis — High overlap]")
    print(f"  가장 어려운 pair: {hp['first']}+{hp['second']} = {hp['accuracy'] * 100:.2f}%")
    print(f"  가장 쉬운 pair:   {ep['first']}+{ep['second']} = {ep['accuracy'] * 100:.2f}%")
    print(
        f"  최저 평균 recall class (전체 test 기준): "
        f"{lr['class']} = {lr['recall'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
