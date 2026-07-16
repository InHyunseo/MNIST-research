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

설치 확인 (설치 후에는 `mnist-overlap` 명령도 동일하게 동작한다):

```bash
python -m mnist_overlap --help
mnist-overlap --help
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
| `dataset` | Canvas, sample 수, stroke threshold |
| `overlap` | 세 overlap 구간과 이동 방향 |
| `model` | Backbone channel과 hidden feature |
| `train` | Epoch, batch, learning rate, early stopping |
| `evaluation` | Batch, hierarchical bootstrap, 신뢰수준, attention 분석 |
| `report` | 결과 파일 형식과 그림 해상도 |

YAML은 로딩 시 `mnist_overlap.config`의 dataclass 스키마로 변환되며 타입과 범위가
검증된다. Class 수(10), MNIST train 총량(60,000), maximum 합성은 config가 아닌 코드
상수다 (구 YAML의 `class_count`, `source_validation_samples`, `composition` 키는 삭제됨).

다른 YAML을 사용하려면 모든 command에 `--config`를 전달한다.

```bash
python -m mnist_overlap train --config configs/mnist_overlap.yaml
```

데이터 생성에 영향을 주는 값을 바꾸면 manifest를 다시 생성해야 한다. 모델 또는 학습
값을 바꾸면 checkpoint를 다시 학습해야 한다. 저장된 fingerprint가 현재 config와
다르면 command가 오래된 artifact 사용을 거부한다.
기본 최종 설정은 training seed `0–9`, 최대 `30 epochs`, patience `3`이다.

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

Best checkpoint에는 best epoch와 validation exact-match 외에 실제 실행 epoch와
`training_complete`가 저장된다. 실행이 중단된 checkpoint는 완료 run으로 인정하지 않으며,
다음 `train` 실행에서 해당 model·seed만 지우고 epoch 1부터 다시 시작한다. Epoch 단위
optimizer resume는 제공하지 않는다.

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

선택한 run의 prediction과 attention cache를 강제로 다시 계산하려면 다음과 같이 실행한다.

```bash
python -m mnist_overlap evaluate --model class_attention --seed 0 --overwrite
```

유효한 NPZ cache가 있으면 inference를 건너뛰고 aggregate CSV만 다시 생성한다. 세 모델과
전체 10개 seed가 모두 준비되면 seed와 pair를 함께 복원추출한 hierarchical bootstrap,
숫자별 recall, seed별 paired 효과와 early-stopping 진단표가 자동 생성된다.

### 저장 결과의 seed 안정성 재분석

이미 생성된 prediction·attention cache와 training log에서 전체 통계와 진단표를 다시
계산할 수 있다. 모델 inference나 재학습은 실행하지 않는다.

```bash
python -m mnist_overlap analyze
```

전체 모델과 seed의 `outputs/predictions/*.npz`, `outputs/attention/*.npz`,
`outputs/training/*.csv`, checkpoint가 모두 필요하다. 전체 `evaluate`를 실행하면 이 분석도
자동으로 함께 생성된다.

### Report

```bash
python -m mnist_overlap report
```

평가 log가 모두 준비된 뒤 실행한다.

## 4. 전체 실험과 결과 생성

다음 script는 데이터 준비, 전체 모델과 seed 학습, 평가, 표·그림 생성을 순서대로
실행한다. 한 번의 실행으로 최종 결과물까지 만드는 기본 명령이다.

처음 10-seed 실험을 시작하면서 기존 artifact를 덮어쓸 때:

```bash
bash scripts/run_all.sh --overwrite
```

중단 후 이어서 실행할 때:

```bash
bash scripts/run_all.sh
```

`--overwrite`는 시작 시 전체 model·seed의 checkpoint, training history, prediction,
attention cache를 한 번 정리한다. 이후 flag 없이 재실행하면 완료 checkpoint와 유효 cache를
재사용한다. 미완료 run은 epoch 1부터 다시 시작하므로 중단 복구 때 `--overwrite`를 다시
붙이지 않는다.

구조 개편 이후 첫 실행은 반드시 `bash scripts/run_all.sh --overwrite`로 시작한다.
설정 fingerprint 산식이 바뀌어 기존 manifest·checkpoint·cache가 모두 무효이기 때문이다.

학습과 평가처럼 시간이 오래 걸리는 계산만 실행해 log를 만들려면 다음 명령을 사용한다.

```bash
python -m mnist_overlap experiment
```

이미 생성된 평가 log에서 표와 그림만 다시 만들려면 다음 명령을 사용한다.

```bash
python -m mnist_overlap report
```

## 5. 결과 위치

```text
outputs/checkpoints/      모델과 seed별 best checkpoint
outputs/training/         Epoch별 학습 이력 CSV
outputs/predictions/      Test logit과 metadata NPZ
outputs/attention/        Sample별 AUPRC, IoU, selectivity, permutation 결과 NPZ
outputs/metrics/          Seed별 metric과 hierarchical 통계 CSV
outputs/experiment_metadata.json  Config snapshot, 환경, 완료 run 목록
results/tables/           최종 비교표
results/figures/          분류 성능, attention behavior, overlap 입력 그림
results/summary.md        핵심 결과 요약
```

10-seed 전체 기준 checkpoint 약 61 MB, prediction cache 약 39 MB와 작은 attention metric
cache·CSV를 포함해 대략 110–130 MB를 예상한다. 전체 attention map은 저장하지 않는다.

## 6. Python 실행 API

외부 application은 필요한 모듈의 공개 실행 함수를 직접 import한다.

```python
from mnist_overlap.config import load_config
from mnist_overlap.data import prepare_data
from mnist_overlap.evaluation import analyze_saved_results, evaluate_models
from mnist_overlap.training import train_models

config = load_config()
prepare_data(config)
train_models()
evaluate_models()           # 유효 cache를 재사용하고 전체 평가 시 통계를 함께 생성
analyze_saved_results()     # 저장된 cache로 통계만 다시 실행할 때 사용
```

CLI도 동일한 공개 함수를 사용한다.

학습된 checkpoint 하나로 단일 이미지를 추론할 때(배포, 향후 ROS2 wrapper)는
`mnist_overlap.inference`만 import하면 된다.

```python
from mnist_overlap.inference import load_model, predict

model = load_model("outputs/checkpoints/class_attention_seed_0.pt")
digit_a, digit_b = predict(model, image)  # image: [76, 76] float array in [0, 1]
```

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

### 실행 중단 후 복구

전체 command를 flag 없이 다시 실행한다. 완료 run은 유지되고 중단된 run만 다시 학습된다.

```bash
bash scripts/run_all.sh
```

### 평가 cache만 다시 만들기

학습 checkpoint는 유지하면서 전체 prediction·attention cache만 다시 계산한다.

```bash
python -m mnist_overlap evaluate --overwrite
python -m mnist_overlap report
```

### CUDA 오류

기본 CPU 실행을 사용하거나 CUDA가 설치된 환경인지 확인한다.

```bash
python -m mnist_overlap train --device cpu
```
