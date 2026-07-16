# 겹친 MNIST 숫자 인식 — MNIST-O

한 장에 겹쳐 배치한 서로 다른 MNIST 숫자 두 개를 동시에 인식하고, 겹침 강도가
LeNet 계열 CNN의 성능에 미치는 영향을 통제된 실험으로 측정한다. Baseline `MnistONet`과
원본 숫자 복원을 보조 과제로 추가한 multitask 모델을 열 개의 같은 학습 seed로 비교한다.

## 문제 정의

두 `28×28` 숫자를 `76×76` 흑백 canvas에 배치하고 같은 위치 pixel의 최댓값으로
합성한다. 정답은 두 숫자 class 위치가 1인 10차원 multi-hot vector다. 모델은 class별
logit 10개를 출력하며, logit이 가장 큰 두 class의 집합이 정답과 정확히 같을 때만
정답으로 판정한다(Top-2 exact-match).

## Controlled Overlap 데이터

겹침 강도는 두 digit bounding box의 교집합 면적을 digit box 면적으로 나눈 값으로
정의한다.

| Level | Bounding-box overlap ratio | 의미 |
|---|---:|---|
| Low | `[0.15, 0.30]` | 두 숫자의 획이 비교적 분리된 조건 |
| Middle | `[0.45, 0.60]` | 일부 획이 섞이는 중간 조건 |
| High | `[0.75, 0.90]` | 두 숫자가 대부분 같은 영역을 차지 |

- 같은 class를 제외한 45개 unordered class pair를 균형화한다.
- 상대 이동은 상·하·좌·우와 네 대각선의 8방향을 사용한다.
- 두 숫자의 pair 중심은 canvas 중앙에 고정하고 변위 크기로만 overlap을 바꾼다.
- MNIST train 원본을 label별로 층화해 train과 validation source를 분리한다.
- Test는 MNIST test split만 사용한다.

Validation과 test에서는 같은 `pair_id`의 두 원본 이미지, class 순서, pair 중심과 이동
방향을 고정하고 변위만 바꿔 Low/Middle/High 표본을 만든다. 따라서 level 간 차이를
paired 비교할 수 있다.

| Split | 규모 |
|---|---:|
| Train | 60,000 images |
| Validation | 3,330 pairs, 9,990 images |
| Test | 10,000 pairs, 30,000 images |

합성 이미지를 저장하는 대신 원본 index와 배치 좌표를 `data/manifests/*.npz`에 기록하고,
Dataset이 실행 중 동일한 이미지를 재구성한다. 데이터 설정의 SHA-256 지문이 달라지면
manifest를 다시 생성한다.

## 모델과 학습

`MnistONet`은 다음 구조로 고정된 LeNet 계열 multi-label 분류기다.

```text
Input 1×76×76
  → Conv(1→6, k5) → ReLU → MaxPool2
  → Conv(6→16, k5) → ReLU → MaxPool2
  → Flatten(4096)
  → FC(4096→120) → ReLU
  → FC(120→84) → ReLU
  → FC(84→10)
```

- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam
- Learning rate: `0.001`
- Batch size: `128`
- 최대 epoch: `30`
- Early stopping: validation Top-2 exact-match, patience 3, minimum delta 0.001
- 학습 seed: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Python·NumPy·PyTorch 난수를 고정하고 PyTorch deterministic algorithm을 강제한다.

Validation exact-match가 가장 높은 checkpoint를 원자적으로 저장한다. 정상 종료된 현재
설정의 checkpoint만 재사용한다.

### Reconstruction multitask

Multitask 모델은 LeNet의 세 spatial feature를 분류 head와 U-Net decoder가 공유한다.
Decoder는 원 [U-Net 논문](https://arxiv.org/abs/1505.04597)의
`up-convolution → encoder feature concat → double convolution` 구조를 두 단계에 적용한다.
Fully connected projection은 사용하지 않는다.

```text
LeNet encoder
  high [6×72×72] ───────────────────────────────┐
  middle [16×32×32] ───────────────┐             │
  bottleneck [16×16×16]             │             │
    → DoubleConv(16→32)              │             │
    → UpConv(32→16) + middle concat ┘             │
    → DoubleConv(32→16)                            │
    → UpConv(16→6) + center-cropped high concat ──┘
    → DoubleConv(12→6) → Conv(6→2, k1) → Sigmoid
    → Spatial source layers [2×64×64]
```

복원 target은 위치가 제거된 원본이 아니라, 입력 canvas와 같은 좌표에 숫자를 각각 배치한
두 source layer의 중앙 `64×64` 영역이다. 현재 overlap 설정의 모든 숫자가 이 영역 안에
완전히 포함된다. 두 출력의 순서가 없으므로 direct/swapped assignment의
intensity-balanced L1을 sample별로 비교하는 PIT loss를 사용한다. 이 loss는 획과 배경의
기여를 각각 정규화해 검은 배경 비율이 학습을 지배하지 않게 한다.

전체 loss는 classification BCE와 reconstruction loss의 가중합이다. Seed 0에서
`λ ∈ {0.05, 0.1, 0.2}`를 validation exact-match로 선택한 뒤, 선택된 λ로 seeds 0–9를
처음부터 공동학습한다. 모든 LeNet parameter는 학습되며 decoder는 추론 시 제거할 수 있다.

## 평가

- Overall/Low/Middle/High별 Top-2 exact-match와 macro-F1의 seed 평균 ± 표본 표준편차
- Baseline은 같은 pair의 Low−High correctness 차이에 대한 bootstrap 신뢰구간
- High overlap에서 가장 어려운/쉬운 숫자 조합
- 전체 test에서 평균 recall이 가장 낮은 class

Baseline 결과는 stdout과 `results/baseline/metrics.json`에 저장한다. 그림은 다음 네 PNG다.

- `training_curves.png`: 10-seed validation accuracy·loss와 epoch별 평균선
- `overlap_examples.png`: 세 숫자 조합의 Low/Middle/High paired 입력 예시
- `overlap_accuracy.png`: overlap별 exact-match seed 평균과 표준편차
- `pair_accuracy_high.png`: High overlap의 숫자 조합별 정확도 heatmap

Multitask 평가는 같은 seed와 test pair에서 baseline을 paired 비교한다. 전체 및 overlap별
정확도 차이의 seed×pair hierarchical bootstrap 구간, 45개 unordered pair confusion,
pair별 accuracy delta를 계산한다. 복원은 spatial layer의 foreground Dice와 balanced L1을
우선 보고하고, 각 정답 위치에서 `28×28`로 자른 복원에 대해서만 L1·MSE·PSNR을 계산한다.
따라서 넓은 검은 배경이 ordinary pixel 지표를 부풀리지 않는다. 결과는
`results/multitask_unet/`에 저장한다.
Multitask 그림에는 같은 형식의 `training_curves.png`와 비교·pair·복원 그림 네 장이
포함된다. 학습 곡선의 accuracy와 loss y축은 모두 `0–1`로 고정한다.

## 설정

`configs/mnist_overlap.yaml`에는 실제 실험에서 조정할 값만 둔다.

```yaml
data:
  seed: 2026
  train_samples: 60000
  validation_pairs: 3330
  test_pairs: 10000

overlap:
  low: [0.15, 0.30]
  middle: [0.45, 0.60]
  high: [0.75, 0.90]

training:
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  maximum_epochs: 30
  batch_size: 128
  learning_rate: 0.001
  early_stopping_patience: 3
  early_stopping_minimum_delta: 0.001

evaluation:
  bootstrap_iterations: 5000
```

Canvas 크기, 모델 layer, MNIST source split, 8방향, 평가 batch size, 95% 신뢰수준과
그림 형식처럼 실험 중 바꾸지 않는 계약은 코드에 고정돼 있다.

`configs/mnist_multitask.yaml`은 baseline config와 reconstruction pilot 범위만 추가한다.

```yaml
baseline_config: configs/mnist_overlap.yaml

reconstruction:
  pilot_seed: 0
  loss_weight_candidates: [0.05, 0.1, 0.2]
```

## 저장소 구조

```text
.
├── README.md
├── main.py                 # baseline·multitask 통합 CLI
├── requirements.txt
├── configs/
│   ├── mnist_overlap.yaml
│   └── mnist_multitask.yaml
├── mnist_overlap/
│   ├── config.py       # 공통 YAML 스키마와 데이터·baseline 경로
│   ├── manifest.py     # source split·manifest 생성과 overlap 좌표 계산
│   ├── data.py         # 합성, 공통 Dataset과 prepare_data 공개 API
│   ├── model.py        # 공통 MnistONet encoder와 분류 head
│   ├── metrics.py      # 공통 분류 지표와 bootstrap
│   ├── runtime.py      # seed, device, DataLoader, 원자 저장
│   ├── main.py         # 기존 baseline 명령 호환 wrapper
│   ├── baseline/       # baseline 학습·추론·평가·그림·main
│   └── multitask/      # decoder·PIT·pilot·비교 평가·그림·main
├── data/               # 자동 생성: MNIST 원본과 manifest
├── outputs/
│   ├── baseline/       # 완료 baseline checkpoint와 학습 이력
│   ├── multitask/      # 이전 FC decoder 실험 보존 경로
│   └── multitask_unet/ # U-Net pilot, final checkpoint와 이력
└── results/
    ├── baseline/       # baseline metrics와 PNG
    ├── multitask/      # 이전 FC decoder 결과 보존 경로
    └── multitask_unet/ # U-Net paired 비교 metrics와 PNG
```

각 모듈은 한 단계의 책임을 맡고 `main.py`가 전체 흐름을 조정한다. 패키지 설치나 별도
shell script 없이 저장소 root에서 바로 실행한다.

## 실행 방법

Python 3.10 이상이 필요하다. 다음 명령으로 `hyunseo-mnist` Conda 가상환경을 만들고
활성화한 뒤 의존성을 설치한다.

```bash
conda create -n hyunseo-mnist python=3.10 -y
conda activate hyunseo-mnist
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

가장 바깥 `main.py`가 권장 통합 진입점이다. 인자 없이 실행하면 baseline을 먼저
완료하거나 기존 checkpoint를 재사용한 뒤 multitask 전체 실험까지 순서대로 실행한다.

```bash
python main.py --device cpu
```

모델 하나만 따로 실행할 수도 있다.

```bash
python main.py --model baseline --device cpu
python main.py --model multitask --device cpu
```

완료 checkpoint로 평가만 다시 수행하거나, 기존 결과로 시각화만 다시 생성한다.

```bash
python main.py --model all --device cpu --skip-training
python main.py --model baseline --device cpu --plot
python main.py --model multitask --device cpu --plot
```

CUDA에서는 cuBLAS 결정론 workspace를 지정한다.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python main.py --model all --device cuda
```

기존 module 명령도 호환성을 위해 유지한다.

```bash
python -m mnist_overlap.main --device cpu
python -m mnist_overlap.baseline.main --device cpu
python -m mnist_overlap.multitask.main --device cpu
```

첫 실행에서는 MNIST를 자동으로 내려받는다. Config가 바뀌면 manifest와 호환되지 않는
checkpoint를 새 설정으로 다시 생성하며, 완료된 호환 checkpoint는 건너뛴다. 이전 FC
decoder 실험의 `outputs/multitask/`와 `results/multitask/`는 U-Net 실행이 덮어쓰지 않는다.

## 해석 범위

결과는 서로 다른 class의 MNIST 숫자 두 개를 `76×76` canvas 중앙에 배치해 maximum으로
합성한 조건에 한정된다. 복원 결과는 입력 좌표의 두 source layer이며, 위치가 제거된
MNIST 원본을 생성하는 과제가 아니다. Maximum 합성에서 소실된 pixel을 완벽히 되찾는다고
주장하지 않으며 분류 encoder의 보조 supervision으로 해석한다. 실제 객체 가림 문제,
다른 데이터셋 또는 다른 모델 구조로의 직접 일반화는 이 실험의 주장 범위가 아니다.
