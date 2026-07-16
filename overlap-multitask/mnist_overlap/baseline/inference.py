"""
추론 단계(`Predictor`)를 정의한다. 각 학습 seed의 best checkpoint를 복원해 test set 전체를
dataset 순서 그대로 추론하고, 이후 평가·시각화가 사용할 클래스별 logit과 pair 메타데이터를
seed별로 모은다. 학습이 끝난 뒤 실행되며 모델 가중치를 바꾸지 않는다.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import CHECKPOINT_DIR, ExperimentConfig
from ..data import ControlledOverlapMnistDataset
from ..model import MnistONet
from .training import load_checkpoint

EVALUATION_BATCH_SIZE = 256
METADATA_FIELD_NAMES = (
    "sample_id",
    "pair_id",
    "label_first",
    "label_second",
    "bounding_box_overlap_ratio",
    "pixel_overlap_ratio",
)


class Predictor:
    """
    입력: test dataset, 학습 seed 목록, seed별 checkpoint
    출력: seed를 key로 하는 prediction dictionary

    각 seed의 best checkpoint로 test prediction을 수집하는 단계이다.
    """

    def __init__(self, config: ExperimentConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    def collect_all_seeds(
        self,
        test_dataset: ControlledOverlapMnistDataset,
        training_seeds: list[int],
    ) -> dict[int, dict[str, np.ndarray]]:
        """
        입력: test_dataset — 추론 대상 test dataset
              training_seeds — 추론할 학습 seed 목록
        출력: seed별 logit·label·metadata array dictionary

        모든 seed의 checkpoint로 test prediction을 수집한다.
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=EVALUATION_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}

        for training_seed in training_seeds:
            model = self._load_model(training_seed)
            predictions_by_seed[training_seed] = self._run_model(model, test_loader)
            print(f"  seed={training_seed} 추론 완료")

        return predictions_by_seed

    def _load_model(self, training_seed: int) -> MnistONet:
        """
        입력: training_seed — 복원할 학습 seed
        출력: 평가 device에 올라간 `MnistONet` 인스턴스

        지정한 seed의 정상 완료 checkpoint를 모델에 복원한다.
        """
        model = MnistONet()
        checkpoint_path = CHECKPOINT_DIR / f"seed_{training_seed}.pt"
        load_checkpoint(model, checkpoint_path, self.device, self.config)

        return model

    @torch.no_grad()
    def _run_model(
        self,
        model: MnistONet,
        test_loader: DataLoader,
    ) -> dict[str, np.ndarray]:
        """
        입력: model — 추론할 모델
              test_loader — dataset 순서를 유지하는 test DataLoader
        출력: logit·multi-hot label·pair metadata array dictionary

        모델 하나로 test loader 전체를 추론해 array로 모은다.
        """
        model.eval()

        collected_columns: dict[str, list[np.ndarray]] = {
            "logits": [],
            "labels": [],
            "overlap_level": [],
        }
        for field_name in METADATA_FIELD_NAMES:
            collected_columns[field_name] = []

        for batch in test_loader:
            logits = model(batch["image"].to(self.device))

            collected_columns["logits"].append(logits.cpu().numpy())
            collected_columns["labels"].append(batch["label"].numpy())
            collected_columns["overlap_level"].append(np.asarray(batch["overlap_level"]))

            for field_name in METADATA_FIELD_NAMES:
                collected_columns[field_name].append(batch[field_name].numpy())

        return {
            field_name: np.concatenate(columns)
            for field_name, columns in collected_columns.items()
        }
