# MNIST Overlap Attention 사용 설명서

## 1. 환경 준비

프로젝트 기준 환경은 Python 3.10, Ubuntu 22.04, CPU용 PyTorch다.

```bash
cd /home/hyunseo/Research/mnist-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

설치 확인:

```bash
python -m mnist_overlap --help
```

Wheel을 다른 위치에 설치해 workspace 밖에서 실행할 때는 runtime root를 명시할 수 있다.

```bash
export MNIST_OVERLAP_ROOT=/home/hyunseo/Research/mnist-project
python -m mnist_overlap --help
```

## 2. Config

기본 설정은 `configs/mnist_overlap.yaml`에 있다.

| Section | 내용 |
|---|---|
| `project` | 데이터 seed와 학습 seed |
| `dataset` | Canvas, sample 수, mask와 합성 방식 |
| `overlap` | 세 overlap 구간과 이동 방향 |
| `model` | Backbone channel과 hidden feature |
| `train` | Epoch, batch, learning rate, early stopping |
| `evaluation` | Batch, bootstrap, attention 분석 |
| `report` | 결과 파일 형식과 그림 해상도 |

다른 YAML을 사용하려면 모든 command에 `--config`를 전달한다.

```bash
python -m mnist_overlap train --config configs/mnist_overlap.yaml
```

데이터 생성에 영향을 주는 값을 바꾸면 manifest를 다시 생성해야 한다. 모델 또는 학습
값을 바꾸면 checkpoint를 다시 학습해야 한다. 저장된 fingerprint가 현재 config와
다르면 command가 오래된 artifact 사용을 거부한다.

## 3. 단계별 실행

### 데이터 준비

```bash
python -m mnist_overlap prepare-data
python -m mnist_overlap validate-data
```

Manifest를 새로 만들 때:

```bash
python -m mnist_overlap prepare-data --overwrite
```

### 모델 학습

전체 모델과 seed:

```bash
python -m mnist_overlap train
```

단일 모델 또는 seed:

```bash
python -m mnist_overlap train --model class_attention --seed 0
```

기존 checkpoint를 다시 학습할 때:

```bash
python -m mnist_overlap train --model class_attention --seed 0 --overwrite
```

CUDA 환경을 명시적으로 사용할 때:

```bash
python -m mnist_overlap train --device cuda
```

### 평가

```bash
python -m mnist_overlap evaluate
```

단일 checkpoint만 확인할 수도 있다.

```bash
python -m mnist_overlap evaluate --model lenet --seed 0
```

세 모델과 전체 seed를 평가해야 bootstrap과 class-pair 비교 CSV가 생성된다.
이때 LeNet의 `Low − High` exact-match 차이와 95% bootstrap interval도 함께 기록된다.

### Report

```bash
python -m mnist_overlap report
```

평가 log가 모두 준비된 뒤 실행한다.

## 4. 전체 실험과 결과 생성

다음 script는 데이터 준비, 전체 모델과 seed 학습, 평가, 표·그림 생성을 순서대로
실행한다. 한 번의 실행으로 최종 결과물까지 만드는 기본 명령이다.

```bash
bash scripts/run_all.sh
```

학습과 평가처럼 시간이 오래 걸리는 계산만 실행해 log를 만들려면 다음 명령을 사용한다.

```bash
bash scripts/run_experiment.sh
```

이미 생성된 평가 log에서 표와 그림만 다시 만들려면 다음 명령을 사용한다.

```bash
bash scripts/run_figures.sh
```

## 5. 결과 위치

```text
models/checkpoints/       모델과 seed별 best checkpoint
logs/training/            Epoch별 학습 이력 CSV
logs/predictions/         Test logit과 metadata NPZ
logs/metrics/             Seed별 metric과 통계 CSV
results/tables/           최종 비교표
results/figures/          Accuracy, heatmap, attention, overlap 입력 그림
results/summary.md        핵심 결과 요약
```

## 6. Python 실행 API

외부 application은 필요한 기능 package의 공개 실행 함수를 직접 import한다.

```python
from mnist_overlap.configuration import load_config
from mnist_overlap.data import prepare_data
from mnist_overlap.evaluation import evaluate_models
from mnist_overlap.training import train_models

config = load_config()
prepare_data(config)
train_models(model_name="lenet", seed=0)
evaluate_models(model_name="lenet", seed=0)
```

CLI도 동일한 공개 함수를 사용한다.

## 7. 문제 해결

### Python package를 찾을 수 없는 경우

가상환경을 활성화하고 editable install을 다시 실행한다.

```bash
source .venv/bin/activate
pip install -e .
```

### Manifest config 오류

```bash
python -m mnist_overlap prepare-data --overwrite
python -m mnist_overlap validate-data
```

### Checkpoint config 오류

해당 모델을 `--overwrite`로 다시 학습한다.

```bash
python -m mnist_overlap train --model lenet --seed 0 --overwrite
```

### CUDA 오류

기본 CPU 실행을 사용하거나 CUDA가 설치된 환경인지 확인한다.

```bash
python -m mnist_overlap train --device cpu
```
