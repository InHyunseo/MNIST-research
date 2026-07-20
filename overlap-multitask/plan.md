# LeNet 분류 + Compact Latent 복원 보조 과제 계획

## 목표

Clipped-sum MNIST-O baseline의 LeNet encoder와 classification head를 유지하고,
class별 compact latent와 shared decoder를 복원 보조 과제로 공동학습한다.

- Baseline: classification `BCEWithLogitsLoss`
- Multitask: classification BCE + `λ_rec × reconstruction loss`
- Pilot: seed 0, `λ_rec ∈ {0.05, 0.1, 0.2}`
- Final: 선택된 λ로 seeds 0–9
- 분류 평가: Overall 및 Low/Middle/High Top-2 exact-match
- 복원 평가: source Dice, balanced BCE, digit-crop L1·MSE·PSNR

## 모델

기존 `MnistONet.layers.*` key와 baseline checkpoint 호환성을 보존한다. 복원 branch는
LeNet convolution bottleneck에 연결하며 U-Net skip connection은 사용하지 않는다.

```text
Input [1×76×76]
  → shared LeNet convolution encoder → bottleneck [16×16×16]
      ├→ 기존 Flatten/FC classifier → logits [10]
      └→ Linear(4096→10×32) + ReLU → class latents [10×32]
          → 요청된 두 class latent 선택 [2×32]
          → class one-hot 결합 [2×42]
          → shared MLP(42→512→1024→4096)
          → source logits [2×64×64]
```

학습·복원 평가에서는 first/second 정답 class를 source 순서대로 제공한다. 일반 추론에서
class를 생략하면 classifier Top-2를 사용한다. 이 구조는 class-conditioned compact
representation을 사용하지만 capsule routing은 구현하지 않으므로 완전한 CapsNet으로
부르지 않는다.

## Target와 loss

두 원본을 입력과 같은 위치의 `[2,64,64]` source map으로 만든다. 출력 순서가 정답
first/second class 조건으로 고정되므로 PIT는 사용하지 않는다.

Reconstruction loss는 다음 두 항을 1:1로 결합한다.

- Balanced BCE: foreground와 background BCE를 각각 정규화해 같은 비중으로 합산
- Source Dice: 두 source map의 `1 − soft Dice` 평균

\[
L=L_{classification}+\lambda_{rec}
\left(0.5L_{balanced\ BCE}+0.5L_{source\ Dice}\right)
\]

한 번의 backward에서 shared convolution encoder는 두 loss의 gradient 합을 받는다.
기존 FC classifier는 분류 loss만, class latent와 shared decoder는 복원 loss만 받는다.

## 평가와 산출물

- Baseline과 multitask의 공통 seeds 0–9 paired 비교
- overlap별 exact-match와 macro-F1 평균±표본 표준편차
- seed×pair hierarchical bootstrap 95% CI
- overlap별 pair accuracy delta와 High pair confusion
- 전체 source map의 Dice와 balanced BCE
- 정답 위치에서 `28×28`로 crop한 L1·MSE·PSNR
- 같은 숫자 pair의 Low/Middle/High reconstruction 예시

새 실험은 기존 semantic U-Net 결과를 덮지 않도록 `outputs/multitask_compact/`와
`results/multitask_compact/`에 저장한다.

## 실행과 검증

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python main.py --model multitask --device cuda
```

검증 조건:

- Baseline split forward, 초기 weight와 checkpoint key가 유지됨
- Class latent가 `[B,10,32]`, 복원 logit이 `[B,2,64,64]`임
- Reconstruction head에 convolution과 skip connection이 없고 decoder가 공유됨
- Dataset target crop이 두 원본 MNIST와 동일함
- 완전한 prediction loss가 blank·혼합·source swap보다 낮음
- Reconstruction/classification gradient 경로가 의도대로 동작함
- Pilot tie-break, fingerprint와 incomplete checkpoint 거부가 동작함
- 단위 테스트와 CPU forward/backward smoke test가 통과함

성능 개선은 구현 완료 조건이 아니다. 개선·무효·악화를 같은 paired 분석으로 보고한다.
