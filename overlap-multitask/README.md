# 겹친 MNIST 숫자 인식 — MNIST-O

한 장에 겹쳐 배치한 서로 다른 MNIST 숫자 두 개를 동시에 인식하고, 겹침 강도가
LeNet 계열 CNN의 성능에 미치는 영향을 통제된 실험으로 측정한다. Baseline `MnistONet`과
원본 숫자 복원을 보조 과제로 추가한 multitask 모델을 열 개의 같은 학습 seed로 비교한다.

## 문제 정의

두 `28×28` 숫자를 `76×76` 흑백 canvas에 배치하고 pixel별로 더한 뒤
`[0,1]` 범위로 clipping해 합성한다. 정답은 두 숫자 class 위치가 1인 10차원
multi-hot vector다. 모델은 class별 logit 10개를 출력하며, logit이 가장 큰 두 class의
집합이 정답과 정확히 같을 때만
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

Multitask 모델은 LeNet convolution encoder를 분류와 복원이 공유하고, bottleneck 뒤에서
두 branch로 나뉜다. 복원 branch는 각 class에 대응하는 32차원 compact latent를 만들고,
정답 두 class의 latent만 하나의 class-conditioned MLP decoder에 통과시킨다. U-Net skip
connection은 사용하지 않는다.

```text
Input → shared LeNet encoder → bottleneck [16×16×16]
                             ├→ 기존 FC head → class logits [10]
                             └→ FC(4096→10×32) → class latents [10×32]
                                  → 두 class latent 선택
                                  → class one-hot 결합 [2×42]
                                  → shared MLP(42→512→1024→4096)
                                  → source logits [2×64×64]
```

학습과 복원 평가는 정답 class를 source 순서대로 latent 선택에 사용한다. Label이 없는
추론에서는 classifier의 Top-2 class를 사용한다. 출력과 target은 first/second source 순서의
`[2,64,64]` map이므로 PIT가 필요 없다. 이 구조는 class-conditioned reconstruction의
아이디어만 사용하며 dynamic routing을 포함한 완전한 CapsNet은 아니다.

Reconstruction loss는 foreground/background를 각각 정규화한 weighted pixel BCE와,
두 source의 soft Dice loss를 1:1로 합친다. 검은 배경의 개수가 loss를 지배하지 않고,
다른 숫자의 획을 함께 출력하면 Dice가 낮아진다.

전체 loss는 classification BCE와 reconstruction loss의 가중합이다. Seed 0에서
`λ ∈ {0.05, 0.1, 0.2}`를 validation exact-match로 선택한 뒤, 선택된 λ로 seeds 0–9를
처음부터 공동학습한다. Shared convolution encoder는 두 loss의 gradient 합을 받고,
classification FC head는 분류 loss만, compact latent와 decoder는 복원 loss만 받는다.

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
pair별 accuracy delta를 계산한다. 복원은 source map의 foreground Dice와 balanced BCE를
우선 보고하고, 각 정답 위치에서 `28×28`로 자른 복원에 대해서만 L1·MSE·PSNR을 계산한다.
따라서 넓은 검은 배경이 ordinary pixel 지표를 부풀리지 않는다. 결과는
`results/multitask_compact/`에 저장한다.
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
│   └── multitask/      # compact latent·loss·pilot·비교 평가·그림·main
├── data/               # 자동 생성: MNIST 원본과 manifest
├── outputs/
│   ├── baseline/       # 완료 baseline checkpoint와 학습 이력
│   └── multitask_compact/ # compact decoder pilot, checkpoint와 이력
└── results/
    ├── baseline/       # baseline metrics와 PNG
    └── multitask_compact/ # compact decoder 비교 metrics와 PNG
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
checkpoint를 새 설정으로 다시 생성하며, 완료된 호환 checkpoint는 건너뛴다.

## 해석 범위

결과는 서로 다른 class의 MNIST 숫자 두 개를 `76×76` canvas 중앙에 배치해
pixel-wise clipped sum으로 합성한 조건에 한정된다. 복원 결과는 입력 좌표의 두 source
map이며, 학습·복원 평가에서는 정답 class identity를 조건으로 제공한다.
Clipped-sum 합성은 포화되지 않은 pixel의 합을 보존하지만, 포화 pixel에서 각 source의
기여도를 유일하게 분리할 수 있다고 주장하지 않는다. 복원은 분류 encoder의 보조
supervision으로 해석한다. 같은 class의
두 instance 분리, 실제 객체 가림 문제와 다른 데이터셋으로 직접 일반화하지 않는다.
