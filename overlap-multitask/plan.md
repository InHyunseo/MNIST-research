# LeNet 분류 + Semantic U-Net 보조 과제 계획

## 목표

기존 LeNet baseline의 encoder와 classification head를 유지하고 U-Net expansive path를
공유 encoder에 연결한다. Decoder는 두 개의 순서 없는 source가 아니라 숫자 class마다
의미가 고정된 10개 spatial map을 예측한다.
입력은 두 source canvas의 pixel-wise 산술평균
`mixed = (canvas_first + canvas_second) / 2`으로 합성한다.

- Baseline: classification `BCEWithLogitsLoss`
- Multitask: classification BCE + `λ_rec × semantic reconstruction loss`
- Pilot: seed 0, `λ_rec ∈ {0.05, 0.1, 0.2}`
- Final: 선택된 λ로 seeds 0–9
- 분류 평가: Overall 및 Low/Middle/High Top-2 exact-match
- 복원 평가: active-class Dice, balanced BCE, digit-crop L1·MSE·PSNR

## 모델

기존 `MnistONet.layers.*` key와 baseline checkpoint 호환성을 보존한다.

```text
Input [1×76×76]
  → Conv5 + ReLU: high [6×72×72]
  → Pool2 → Conv5 + ReLU: middle [16×32×32]
  → Pool2: bottleneck [16×16×16]

Classification
  bottleneck → 기존 Flatten/FC head → logits [10]

Semantic reconstruction
  bottleneck → DoubleConv(16→32)
  → UpConv(32→16) + middle concat → DoubleConv(32→16)
  → UpConv(16→6) + cropped high concat → DoubleConv(12→6)
  → Conv1×1(6→10) → semantic logits [10×64×64]
```

Decoder에는 `Flatten`과 `Linear`를 사용하지 않는다. 원 U-Net처럼 up-convolution,
동일 scale encoder feature concat, double `3×3` convolution과 마지막 `1×1` convolution을
사용한다.

## Target와 loss

숫자 `k`의 source image를 입력과 같은 offset으로 class channel `k`에 배치한다. 각 sample은
서로 다른 두 class만 사용하므로 target `[10,64,64]`에서 두 channel만 활성화된다. 겹친
pixel은 두 channel에 동시에 존재할 수 있어 softmax가 아닌 독립 sigmoid를 사용한다.

출력 channel 의미가 class로 고정되므로 PIT와 direct/swapped assignment를 제거한다.
Reconstruction loss는 다음 두 항을 1:1로 결합한다.

- Balanced pixel BCE: foreground와 background BCE를 각각 정규화해 같은 비중으로 합산
- Active Dice loss: 실제 숫자가 존재하는 두 class map의 `1 − soft Dice`

전체 loss는 다음과 같다.

\[
L=L_{classification}+\lambda_{rec}
\left(0.5L_{balanced\ BCE}+0.5L_{active\ Dice}\right)
\]

## 평가와 시각화

- Baseline과 multitask의 공통 seeds 0–9 paired 비교
- overlap별 exact-match와 macro-F1 평균±표본 표준편차
- seed×pair hierarchical bootstrap 95% CI
- overlap별 pair accuracy delta와 High pair confusion
- 전체 semantic map의 active-class Dice와 balanced BCE
- 정답 class channel을 source offset에서 `28×28`로 crop한 L1·MSE·PSNR
- 같은 3+8 pair의 Low/Middle/High class-specific reconstruction 예시

학습 곡선과 기존 비교 그림 다섯 장을 `results/multitask_unet/figures/`에 저장한다.

## 실행과 검증

기존 통합 CLI를 유지한다. Composition mode를 data·model fingerprint에 포함하므로
이전 maximum 합성 checkpoint는 재사용하지 않는다. Baseline과 multitask를 모두
같은 mean 합성 데이터로 다시 학습하고, 기존 각 output 경로의 비호환 파일을 교체한다.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python main.py --model all --device cuda
```

검증 조건:

- Baseline split forward와 checkpoint key가 유지됨
- Decoder가 fully convolutional이고 `[B,10,64,64]` logit을 출력함
- Dataset target에서 정답 class 두 channel만 활성화됨
- Class channel에서 crop한 target이 두 원본 MNIST와 동일함
- 완전한 semantic prediction loss가 blank·혼합·class swap보다 낮음
- Reconstruction/classification gradient 경로가 올바르게 분리됨
- Pilot tie-break, fingerprint와 incomplete checkpoint 거부가 동작함
- 전체 unit test와 CPU forward/backward smoke test가 통과함

성능 개선은 완료 조건이 아니다. 개선·무효·악화를 같은 paired 분석으로 보고한다.
