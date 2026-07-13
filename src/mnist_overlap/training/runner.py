"""선택한 모델과 seed 조합의 학습 실행을 조정한다.

입력:
    Config 경로, model/seed 선택, device, checkpoint 덮어쓰기 여부

출력:
    각 실행의 TrainingResult 목록

연결:
    CLI와 외부 application이 training package의 공개 함수로 호출한다.
"""

from pathlib import Path

import torch

from ..configuration import (
    CHECKPOINT_DIR,
    DEFAULT_CONFIG_PATH,
    TRAINING_LOG_DIR,
    create_output_directories,
    load_config,
)
from ..data import ControlledOverlapMnistDataset
from ..models import create_model
from .engine import (
    TrainingResult,
    checkpoint_matches_config,
    set_random_seed,
    train_model,
)


def train_models(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_name: str | None = None,
    seed: int | None = None,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> list[TrainingResult]:
    """Config에 지정된 모델과 seed 또는 선택한 단일 조합을 학습한다.

    입력:
        Config 경로, 선택 model/seed, `cpu` 또는 `cuda`, overwrite option

    처리:
        동일 data order와 초기화 seed를 적용하고 각 모델의 best checkpoint를 저장한다.

    출력:
        실제로 새로 학습한 실행의 TrainingResult 목록
    """
    config = load_config(config_path)
    create_output_directories()
    device = _select_device(device_name)
    model_names = _select_model_names(config, model_name)
    seeds = _select_seeds(config, seed)

    training_config = config["train"]
    train_dataset = ControlledOverlapMnistDataset("train", config)
    validation_dataset = ControlledOverlapMnistDataset("validation", config)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["data_loader_workers"]),
    )

    training_results = []

    for selected_seed in seeds:
        for selected_model_name in model_names:
            checkpoint_path = (
                CHECKPOINT_DIR / f"{selected_model_name}_seed_{selected_seed}.pt"
            )
            history_path = (
                TRAINING_LOG_DIR / f"{selected_model_name}_seed_{selected_seed}.csv"
            )

            if checkpoint_path.exists() and not overwrite:
                if checkpoint_matches_config(checkpoint_path, config):
                    print(f"기존 checkpoint를 사용합니다: {checkpoint_path}")
                    continue

                raise RuntimeError(
                    f"현재 config와 다른 checkpoint입니다: {checkpoint_path}. "
                    "--overwrite option으로 다시 학습하세요."
                )

            set_random_seed(selected_seed)
            train_loader = _create_train_loader(
                train_dataset,
                training_config,
                selected_seed,
            )
            model = create_model(selected_model_name, config)
            result = train_model(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                config=config,
                model_name=selected_model_name,
                seed=selected_seed,
                checkpoint_path=checkpoint_path,
                history_path=history_path,
                device=device,
            )
            training_results.append(result)

    return training_results


def _select_device(device_name: str) -> torch.device:
    """요청한 학습 device가 사용 가능한지 확인한다.

    입력:
        `cpu` 또는 `cuda` 문자열

    처리:
        CUDA 요청 시 PyTorch 가용성을 검사한다.

    출력:
        검증된 `torch.device`
    """
    if device_name not in ("cpu", "cuda"):
        raise ValueError("Device는 'cpu' 또는 'cuda'여야 합니다.")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 현재 환경에서 사용할 수 없습니다.")

    return torch.device(device_name)


def _select_model_names(
    config: dict,
    model_name: str | None,
) -> list[str]:
    """단일 model option 또는 config의 전체 model 목록을 선택한다.

    입력:
        전체 config와 선택적인 model 이름

    처리:
        이름이 없으면 config 순서를 유지하고 있으면 지원 목록인지 확인한다.

    출력:
        학습할 model 이름 목록
    """
    configured_names = list(config["model"]["model_names"])

    if model_name is None:
        return configured_names

    if model_name not in configured_names:
        raise ValueError(f"Config에 없는 모델입니다: {model_name}")

    return [model_name]


def _select_seeds(config: dict, seed: int | None) -> list[int]:
    """단일 seed option 또는 config의 전체 seed 목록을 선택한다.

    입력:
        전체 config와 선택적인 정수 seed

    처리:
        Seed가 없으면 config 값을 읽고 모든 값을 정수로 변환한다.

    출력:
        학습할 정수 seed 목록
    """
    if seed is not None:
        return [int(seed)]

    return [int(configured_seed) for configured_seed in config["project"]["training_seeds"]]


def _create_train_loader(
    train_dataset: ControlledOverlapMnistDataset,
    training_config: dict,
    seed: int,
) -> torch.utils.data.DataLoader:
    """모델 간 동일한 batch 순서를 재현하는 train DataLoader를 생성한다.

    입력:
        Train Dataset, training section, 현재 seed

    처리:
        별도 torch.Generator에 seed를 적용하고 shuffle을 활성화한다.

    출력:
        재현 가능한 train DataLoader
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["data_loader_workers"]),
        generator=generator,
    )
