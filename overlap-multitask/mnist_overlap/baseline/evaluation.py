"""
평가 단계(`Evaluator`)를 정의한다. 여러 seed의 test 추론 결과에서 overlap level별 분류
성능, 겹침이 강해질 때의 성능 저하(Low−High) 핵심 효과와 그 hierarchical bootstrap
신뢰구간, seed별 효과의 부호 안정성, High overlap에서의 failure 분석(가장 어렵/쉬운 숫자
조합·최저 recall class)을 계산한다. 결과를 stdout으로 보고하고
`results/baseline/metrics.json`으로 저장한다.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import torch

from ..config import CLASS_COUNT, RESULTS_DIR, ExperimentConfig
from ..metrics import (
    class_pair_accuracy,
    classification_metrics,
    exact_match_per_sample,
    hierarchical_bootstrap_interval,
    paired_bootstrap_interval,
    paired_level_difference,
    sample_deviation,
)

OVERLAP_LEVELS = ("low", "middle", "high")
METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"
CONFIDENCE_LEVEL = 0.95


class Evaluator:
    """
    입력: predictions_by_seed, training_seeds, ExperimentConfig
    출력: metrics dictionary, results/baseline/metrics.json, stdout 요약

    test prediction에서 발표에 필요한 수치를 계산하고 저장·출력하는 단계이다.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    # -------------------------------------------------------------------------
    # 지표 계산
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        predictions_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> dict:
        """
        입력: predictions_by_seed — seed별 test prediction
              training_seeds — 학습 seed 목록
        출력: performance·headline_low_minus_high·per_seed_low_minus_high·
              failure_analysis_high를 담은 metrics dictionary

        발표에 필요한 모든 수치를 하나의 dictionary로 계산한다.
        """
        reference = predictions_by_seed[training_seeds[0]]

        iterations = self.config.evaluation.bootstrap_iterations
        confidence_level = CONFIDENCE_LEVEL
        bootstrap_seed = self.config.data.seed

        performance = self._compute_performance(predictions_by_seed, training_seeds)

        correctness_by_seed = {
            training_seed: self._correct_per_sample(predictions_by_seed[training_seed])
            for training_seed in training_seeds
        }

        low_minus_high_matrix = np.stack([
            paired_level_difference(
                correctness_by_seed[training_seed],
                reference["pair_id"],
                reference["overlap_level"],
                "low",
                "high",
            )[0]
            for training_seed in training_seeds
        ])
        headline_estimate, headline_lower, headline_upper = hierarchical_bootstrap_interval(
            low_minus_high_matrix, iterations, confidence_level, bootstrap_seed
        )

        per_seed_low_minus_high = self._compute_per_seed_low_minus_high(
            correctness_by_seed, reference, training_seeds
        )
        failure_analysis = self._compute_failure_analysis(predictions_by_seed, training_seeds)

        return {
            "model": "MnistONet",
            "training_seeds": training_seeds,
            "bootstrap_iterations": iterations,
            "confidence_level": confidence_level,
            "performance": performance,
            "headline_low_minus_high": {
                "estimate": headline_estimate,
                "confidence_lower": headline_lower,
                "confidence_upper": headline_upper,
                "seed_count": int(low_minus_high_matrix.shape[0]),
                "pair_count": int(low_minus_high_matrix.shape[1]),
            },
            "per_seed_low_minus_high": per_seed_low_minus_high,
            "failure_analysis_high": failure_analysis,
        }

    def _compute_performance(
        self,
        predictions_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> dict[str, dict[str, float]]:
        """
        입력: predictions_by_seed — seed별 test prediction
              training_seeds — 학습 seed 목록
        출력: overlap level("all"·low·middle·high)별 평균·표본 표준편차 dictionary

        overlap level별 exact-match와 macro-F1의 seed 평균·표본 표준편차를 계산한다.
        """
        performance: dict[str, dict[str, float]] = {}

        for level in ("all", *OVERLAP_LEVELS):
            exact_match_values = []
            macro_f1_values = []

            for training_seed in training_seeds:
                predictions = predictions_by_seed[training_seed]

                level_mask = (
                    np.ones(len(predictions["labels"]), dtype=bool)
                    if level == "all"
                    else predictions["overlap_level"] == level
                )
                metrics = classification_metrics(
                    predictions["logits"][level_mask],
                    predictions["labels"][level_mask],
                )

                exact_match_values.append(metrics["exact_match"])
                macro_f1_values.append(metrics["macro_f1"])

            exact_match_values = np.asarray(exact_match_values)
            macro_f1_values = np.asarray(macro_f1_values)

            performance[level] = {
                "exact_match_mean": float(exact_match_values.mean()),
                "exact_match_standard_deviation": sample_deviation(exact_match_values),
                "macro_f1_mean": float(macro_f1_values.mean()),
                "macro_f1_standard_deviation": sample_deviation(macro_f1_values),
            }

        return performance

    def _compute_per_seed_low_minus_high(
        self,
        correctness_by_seed: dict[int, np.ndarray],
        reference: dict[str, np.ndarray],
        training_seeds: list[int],
    ) -> list[dict]:
        """
        입력: correctness_by_seed — seed별 sample correctness 배열
              reference — pair_id·overlap_level을 제공하는 기준 prediction
              training_seeds — 학습 seed 목록
        출력: seed별 추정값·신뢰구간 dictionary 목록

        seed별 Low − High 효과와 test-pair bootstrap 구간을 계산한다.
        """
        iterations = self.config.evaluation.bootstrap_iterations
        confidence_level = CONFIDENCE_LEVEL
        bootstrap_seed = self.config.data.seed

        results = []

        for offset, training_seed in enumerate(training_seeds):
            differences, pair_ids = paired_level_difference(
                correctness_by_seed[training_seed],
                reference["pair_id"],
                reference["overlap_level"],
                "low",
                "high",
            )
            estimate, lower, upper = paired_bootstrap_interval(
                differences, pair_ids, iterations, confidence_level, bootstrap_seed + offset
            )

            results.append({
                "seed": training_seed,
                "estimate": estimate,
                "confidence_lower": lower,
                "confidence_upper": upper,
            })

        return results

    def _compute_failure_analysis(
        self,
        predictions_by_seed: dict[int, dict[str, np.ndarray]],
        training_seeds: list[int],
    ) -> dict:
        """
        입력: predictions_by_seed — seed별 test prediction
              training_seeds — 학습 seed 목록
        출력: hardest_pair·easiest_pair·lowest_recall_class_overall dictionary

        High overlap에서 가장 어렵/쉬운 class pair와 전체 test 최저 recall class를 찾는다.
        """
        reference = predictions_by_seed[training_seeds[0]]

        high_mask = reference["overlap_level"] == "high"
        label_first_high = reference["label_first"][high_mask]
        label_second_high = reference["label_second"][high_mask]

        pair_accuracy_matrices = []
        recall_vectors = []

        for training_seed in training_seeds:
            predictions = predictions_by_seed[training_seed]
            correctness = self._correct_per_sample(predictions)

            pair_accuracy_matrices.append(
                class_pair_accuracy(
                    correctness[high_mask], label_first_high, label_second_high, CLASS_COUNT
                )
            )
            recall_vectors.append(
                classification_metrics(predictions["logits"], predictions["labels"])[
                    "per_class_recall"
                ]
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_pair_accuracy = np.nanmean(np.stack(pair_accuracy_matrices), axis=0)

        mean_recall = np.stack(recall_vectors).mean(axis=0)

        pair_entries = [
            (first_class, second_class, float(mean_pair_accuracy[first_class, second_class]))
            for first_class in range(CLASS_COUNT)
            for second_class in range(first_class + 1, CLASS_COUNT)
            if np.isfinite(mean_pair_accuracy[first_class, second_class])
        ]
        hardest_pair = min(pair_entries, key=lambda entry: entry[2])
        easiest_pair = max(pair_entries, key=lambda entry: entry[2])
        lowest_recall_class = int(np.argmin(mean_recall))

        return {
            "hardest_pair": {
                "first": hardest_pair[0],
                "second": hardest_pair[1],
                "accuracy": hardest_pair[2],
            },
            "easiest_pair": {
                "first": easiest_pair[0],
                "second": easiest_pair[1],
                "accuracy": easiest_pair[2],
            },
            "lowest_recall_class_overall": {
                "class": lowest_recall_class,
                "recall": float(mean_recall[lowest_recall_class]),
            },
        }

    def _correct_per_sample(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """
        입력: predictions — "logits"·"labels" array를 담은 dictionary
        출력: sample별 정답 여부 float 배열 (1.0 정답 / 0.0 오답)

        Top-2 예측이 정답과 정확히 일치하는지 sample별로 판정한다.
        """
        correctness = exact_match_per_sample(
            torch.from_numpy(predictions["logits"]),
            torch.from_numpy(predictions["labels"]),
        )

        return correctness.numpy().astype(np.float64)

    # -------------------------------------------------------------------------
    # 출력과 저장
    # -------------------------------------------------------------------------

    def report(self, metrics: dict) -> None:
        """
        입력: metrics — `evaluate`가 반환한 수치 dictionary
        출력: 없음 (stdout 출력)

        계산한 수치를 발표 슬롯 순서대로 stdout에 정리한다.
        """
        confidence_percent = metrics["confidence_level"] * 100.0
        performance = metrics["performance"]

        print("\n===== MNIST-O 결과 요약 =====")

        print("\n[Top-2 exact-match accuracy (%)] seed 평균 ± 표본 std")
        for level in ("all", "low", "middle", "high"):
            entry = performance[level]
            print(
                f"  {level:8} {entry['exact_match_mean'] * 100:6.2f} ± "
                f"{entry['exact_match_standard_deviation'] * 100:.2f}"
            )

        print("\n[Macro-F1] seed 평균 ± 표본 std")
        for level in ("all", "low", "middle", "high"):
            entry = performance[level]
            print(
                f"  {level:8} {entry['macro_f1_mean']:.4f} ± "
                f"{entry['macro_f1_standard_deviation']:.4f}"
            )

        headline = metrics["headline_low_minus_high"]
        print(
            f"\n[Headline] Low − High = {headline['estimate'] * 100:.2f}%p, "
            f"{confidence_percent:.0f}% CI "
            f"[{headline['confidence_lower'] * 100:.2f}, {headline['confidence_upper'] * 100:.2f}] "
            f"(seed+pair hierarchical bootstrap, {metrics['bootstrap_iterations']:,}회)"
        )

        per_seed_estimates = [row["estimate"] for row in metrics["per_seed_low_minus_high"]]
        positive_count = sum(1 for estimate in per_seed_estimates if estimate > 0)
        print(
            f"  seed별 Low − High: {positive_count}/{len(per_seed_estimates)} 양수 "
            f"(범위 {min(per_seed_estimates) * 100:.2f} ~ {max(per_seed_estimates) * 100:.2f}%p)"
        )

        failure = metrics["failure_analysis_high"]
        hardest_pair = failure["hardest_pair"]
        easiest_pair = failure["easiest_pair"]
        lowest_recall = failure["lowest_recall_class_overall"]

        print("\n[Failure analysis — High overlap]")
        print(
            f"  가장 어려운 pair: {hardest_pair['first']}+{hardest_pair['second']} = "
            f"{hardest_pair['accuracy'] * 100:.2f}%"
        )
        print(
            f"  가장 쉬운 pair:   {easiest_pair['first']}+{easiest_pair['second']} = "
            f"{easiest_pair['accuracy'] * 100:.2f}%"
        )
        print(
            f"  최저 평균 recall class (전체 test 기준): "
            f"{lowest_recall['class']} = {lowest_recall['recall'] * 100:.2f}%"
        )

    def save(self, metrics: dict) -> None:
        """
        입력: metrics — 저장할 수치 dictionary
        출력: 없음 (results/baseline/metrics.json 파일 생성)

        수치 dictionary를 JSON 파일로 저장한다.
        """
        METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_JSON_PATH.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n수치를 저장했습니다: {METRICS_JSON_PATH}")
