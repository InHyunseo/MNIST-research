# LeNet 분류 + U-Net 복원 보조 과제 구현 계획

## 1. 목표

기존 LeNet baseline의 분류 구조와 checkpoint를 그대로 보존하면서 convolution encoder에
U-Net expansive path를 연결한다. 두 숫자를 입력 canvas 좌표에서 분리하는 복원 보조
과제가 Top-2 분류 성능에 주는 영향을 baseline과 같은 10 seeds로 비교한다.

- Baseline: `BCEWithLogitsLoss`
- Multitask: `BCEWithLogitsLoss + λ_rec × balanced PIT-L1`
- Pilot: seed 0, `λ_rec ∈ {0.05, 0.1, 0.2}`
- 본 실험: 선택된 λ로 seeds 0–9
- 주 분류 지표: Overall 및 Low/Middle/High Top-2 exact-match
- 주 복원 지표: foreground Dice와 balanced L1
- 기존 FC decoder 실험 결과는 별도 경로에 보존

## 2. U-Net 모델 계약

원 U-Net의 다음 원칙을 그대로 따른다.

- Decoder에는 `Flatten`과 fully connected layer를 사용하지 않는다.
- 각 단계에서 up-convolution으로 해상도를 두 배로 높인다.
- 같은 해상도의 encoder feature를 중앙 crop한 뒤 channel 방향으로 concat한다.
- concat 뒤 `3×3 convolution + ReLU`를 두 번 적용한다.
- 마지막 `1×1` convolution으로 두 source channel을 출력한다.

기존 LeNet parameter key를 유지하기 위해 contracting path는 바꾸지 않는다.

```text
Input [1×76×76]
  → Conv5 + ReLU: high [6×72×72]
  → Pool2 → Conv5 + ReLU: middle [16×32×32]
  → Pool2: bottleneck [16×16×16]

Classification
  bottleneck → 기존 Flatten/FC head → logits [10]

Reconstruction
  bottleneck → DoubleConv(16→32)
  → UpConv(32→16) + middle concat → DoubleConv(32→16)
  → UpConv(16→6) + center-cropped high concat → DoubleConv(12→6)
  → Conv1×1(6→2) → Sigmoid → source layers [2×64×64]
```

`MnistONet.encode_with_skips()`는 `[6,72,72]`, `[16,32,32]`, `[16,16,16]`
feature를 반환한다. 기존 `encode()`와 baseline forward 결과는 동일하게 유지한다.

## 3. Spatial target와 loss

U-Net은 입력과 출력의 좌표가 대응하는 localization 모델이다. 따라서 위치가 제거된
`28×28` MNIST 원본이 아니라, 두 원본을 입력과 같은 offset에 각각 배치한 source layer를
target으로 사용한다. LeNet decoder 출력과 정확히 맞는 canvas 중앙 `[6:70, 6:70]`의
`64×64` 영역을 사용한다. 현재 manifest의 모든 digit extent `[13:64)`가 이 안에 들어간다.

두 output channel의 순서는 없으므로 direct/swapped assignment의 intensity-balanced L1을
sample별로 비교한다. Foreground와 background error를 따로 정규화해 1:1로 합치므로 검은
배경 pixel 수가 loss를 지배하지 않는다. Classification과 reconstruction loss는 모든
LeNet encoder parameter를 공동으로 업데이트하며 classification head와 decoder는 각자의
loss에서만 gradient를 받는다.

## 4. 평가와 그림

분류는 기존 paired 분석을 유지한다.

- 모델별 Overall/Low/Middle/High exact-match와 macro-F1 평균±표본 표준편차
- `Multitask − Baseline` exact-match의 seed×pair hierarchical bootstrap 95% CI
- overlap별 45 pair accuracy와 delta matrix
- High overlap의 normalized 45×45 pair confusion

복원 평가는 배경으로 수치가 좋아 보이는 문제를 피하도록 구분한다.

- 전체 spatial source layer: PIT foreground Dice, balanced L1
- 각 정답 offset에서 자른 `28×28` digit: L1, MSE, PSNR
- 복원 그림: Mixed, 두 `28×28` GT, 같은 위치에서 자른 두 PIT-matched 복원

`training_curves.png`, `overlap_comparison.png`, `pair_accuracy_difference.png`,
`pair_confusion_high.png`, `reconstruction_examples.png`를 생성한다. 제목은 두 단어 이내,
불필요한 label은 제거하고 학습 accuracy와 loss y축은 `0–1`로 고정한다.

## 5. 산출물과 실행

기존 FC decoder 실험을 덮어쓰지 않는다.

```text
outputs/multitask/       # 이전 FC decoder checkpoint·log
results/multitask/       # 이전 FC decoder metrics·figure
outputs/multitask_unet/  # 새 U-Net pilot·checkpoint·log
results/multitask_unet/  # 새 U-Net metrics·figure
```

기존 CLI를 그대로 사용한다.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python main.py --model multitask --device cuda

python main.py --model multitask --device cuda --skip-training
python main.py --model multitask --device cuda --plot
```

## 6. 검증 기준

- 기존 baseline checkpoint key와 forward output이 유지됨
- Decoder에 `Flatten`과 `Linear`가 없음
- 세 encoder feature와 두 skip의 spatial shape가 계약과 일치함
- 출력이 `[B,2,64,64]`, 값 범위가 `[0,1]`임
- Target의 `28×28` source crop이 원본과 pixel 단위로 동일함
- PIT loss가 output channel 교환에 불변이고 blank보다 정답 loss가 낮음
- 정답의 foreground Dice가 1이고 blank Dice보다 높음
- Reconstruction-only/classification-only gradient 경로가 분리됨
- 같은 seed의 baseline과 multitask 초기 LeNet state가 동일함
- 이전 FC 산출물과 새 U-Net 산출물 경로가 격리됨
- Pilot tie-break, fingerprint, incomplete checkpoint 거부가 동작함
- 전체 unit test와 작은 CPU forward/backward smoke test가 통과함

성능 개선은 구현 완료 조건이 아니다. 개선·무효·악화를 같은 paired 분석으로 보고한다.
