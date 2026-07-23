# Denoising-Auxiliary Classification on n-MNIST

노이즈가 포함된 MNIST 분류에서 clean image 복원을 보조 과제로 함께 학습하면
분류 정확도가 향상되는지 확인한다. Baseline과 multitask 모델은 동일한 LeNet encoder와
classification head를 사용하며, multitask 학습에만 denoising decoder와 MSE loss가
추가된다. 추론 시에는 두 모델 모두 classification 경로만 사용한다.

## 데이터

[n-MNIST 배포본](https://www.csc.lsu.edu/~saikat/n-mnist/)의 세 noise 조건을 그대로
사용한다. 각 조건은 MNIST와 동일하게 training image 60,000장과 test image 10,000장으로
구성되며, noise의 종류와 강도는 배포본에 고정돼 있다.

| Noise | 배포본 설정 |
|---|---|
| AWGN | SNR 9.5 |
| Motion blur | 이동 거리 5px, 반시계 방향 15° |
| Reduced contrast + AWGN | Contrast 50%, SNR 12 |

각 seed에서 training 60,000장을 54,000장 학습용과 6,000장 validation용으로 나눈다.
Test 10,000장은 checkpoint 선택에 사용하지 않고 최종 평가에만 사용한다. 입력과 clean
target은 같은 index의 이미지이며, `28×28` 영상을 양쪽에 2px씩 padding해 `1×32×32`로
모델에 전달한다.

## 모델과 손실함수

Classification 경로는 PyTorch의 공식 `Neural Networks` tutorial에 제시된 LeNet 구현과
동일하다.

```text
Input 1×32×32
  → Conv(1→6, 5×5) → ReLU → MaxPool(2×2)
  → Conv(6→16, 5×5) → ReLU → MaxPool(2×2)
  → Flatten(400)
  → FC(400→120) → ReLU
  → FC(120→84) → ReLU
  → FC(84→10)
```

이는 원본 LeNet-5(1998)의 activation과 subsampling을 문자 그대로 재현한 구조가 아니라,
PyTorch tutorial의 현대적 LeNet 구현이다. Multitask 모델은 shared encoder의
`16×5×5` feature에서 다음 decoder로 분기한다.

```text
16×5×5 → ConvTranspose(16→6, 6×6, stride 2) → ReLU
       → ConvTranspose(6→1, 6×6, stride 2) → 1×32×32
```

Baseline은 cross-entropy만 최소화한다. Multitask 모델의 손실은 다음과 같다.

```text
L = L_CE + λ × L_MSE
```

각 noise의 seed 0에서 `λ ∈ {0.05, 0.10, 0.20}`을 비교하고 가장 높은 validation
classification accuracy를 얻은 값을 선택했다. 동률이면 더 작은 λ를 사용한다.

| Noise | 선택된 λ |
|---|---:|
| AWGN | 0.05 |
| Motion blur | 0.10 |
| Reduced contrast + AWGN | 0.10 |

## 실험 설정

| 항목 | 설정 |
|---|---|
| Seeds | 0–29, paired between conditions |
| Optimizer | Adam, β₁=0.9, β₂=0.999 |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 30 |
| Validation ratio | 0.1 |
| AMP | 사용하지 않음 |
| Checkpoint | 최고 validation classification accuracy epoch |

모든 seed에서 baseline과 multitask가 같은 data split을 사용한다. Multitask 모델의
decoder는 학습 때만 실행하며 test classification과 실제 추론에서는 건너뛴다.

## 결과

아래 값은 30개 seed의 test accuracy 평균 ± 표본 표준편차다. Δ는 같은 seed의
`multitask − baseline` 정확도 차이를 percentage point 단위로 계산한 값이며, 신뢰구간과
p-value는 paired t-test 기준이다.

| Noise | Baseline | Multitask | Δ accuracy | 95% CI | p-value |
|---|---:|---:|---:|---:|---:|
| AWGN | 98.184 ± 0.121% | 98.186 ± 0.116% | +0.002 pp | [-0.052, +0.057] | 0.9305 |
| Motion blur | 98.836 ± 0.102% | 98.898 ± 0.073% | +0.062 pp | [+0.024, +0.101] | 0.0026 |
| Reduced contrast + AWGN | 96.745 ± 0.158% | 96.770 ± 0.201% | +0.025 pp | [-0.052, +0.102] | 0.5116 |

복원 보조 과제는 Motion blur에서만 통계적으로 유의한 분류 성능 향상을 보였다. AWGN과
Reduced contrast + AWGN에서는 평균 정확도가 소폭 증가했지만 신뢰구간이 0을 포함하므로
일관된 개선으로 해석하지 않는다. 따라서 clean reconstruction이 모든 noise 조건에
보편적으로 유효하다고 결론내릴 수는 없다.

## 후속 분석: 학습 중 gradient alignment

최종 checkpoint 한 시점의 gradient만으로는 학습 과정을 설명할 수 없으므로, 별도
명령에서 baseline과 multitask를 같은 설정으로 학습하며 multitask의 매 epoch gradient를
기록한다. 이미 선택한 λ(AWGN 0.05, 나머지 0.10)를 그대로 사용하므로 pilot은 반복하지
않는다.
각 seed의 고정된 validation 8개 batch에서 classification loss와 reconstruction loss가
shared encoder에 주는 gradient를 각각 계산한다. Probe 계산은 parameter를 업데이트하지
않는다.

```text
cosine = (g_CE · g_MSE) / (‖g_CE‖ ‖g_MSE‖)
relative norm = ‖λg_MSE‖ / ‖g_CE‖
```

Cosine이 양수면 두 loss가 encoder를 비슷한 방향으로, 음수면 충돌하는 방향으로
학습시키는 것으로 해석한다. Relative norm은 방향과 별개로 reconstruction gradient가
실제 joint loss에서 차지하는 상대적 크기를 나타낸다. 분석은 early epoch 1–10, middle
11–20, late 21–30의 변화와 seed별 mean cosine–accuracy delta 상관을 함께 출력한다.
이는 gradient 정렬과 성능 향상의 연관성을 확인하는 분석이며 인과관계를 단독으로
증명하지는 않는다.

## 실행

Python 3.10 이상 환경에서 dependency를 설치하고 아래 진입점만 사용한다.

```bash
python -m pip install -r requirements.txt

python main.py data
python main.py train-baseline --device cuda
python main.py train-multitask --device cuda
python main.py plot
```

빈 `outputs/`에서 baseline, gradient 측정을 포함한 multitask, 통계 요약까지 한 번에
실행하려면 다음 명령을 사용한다.

```bash
python main.py train-alignment --device cuda
```

30개 seed를 사용하며, 중단 후 같은 명령을 실행하면 checkpoint와 gradient CSV가 모두
정상 저장된 run은 건너뛴다. Epoch별 측정값은
`outputs/gradient_alignment/measurements.csv`, 노이즈별 통계는 `summary.csv`에 남는다.

Motion Blur에서 더 큰 λ `1`, `3`, `10`을 각각 5개 seed로 비교하려면 다음 명령을
사용한다. 기존 λ `0.1` 실험은 다시 실행하지 않으며, 결과 요약은
`outputs/gradient_alignment/weight_sweep_summary.csv`에 저장한다.

```bash
python main.py train-lambda-sweep --device cuda
```

`--device`는 `auto`, `cpu`, `cuda`를 지원하며 기본값은 `auto`다. `plot`은 학습을
실행하지 않고 준비된 data와 저장된 checkpoint, history 및 CSV 결과만 읽는다.

## 파일 구조

```text
denoising-multitask/
├── data/
│   ├── raw/                 n-MNIST archive와 MAT
│   └── mnist/               index가 대응되는 clean MNIST MAT
├── outputs/
│   ├── checkpoints/         final run별 best checkpoint
│   ├── figures/             발표와 분석에 사용하는 PNG
│   ├── gradient_alignment/
│   │   ├── measurements.csv 학습 중 epoch별 gradient 측정값
│   │   ├── summary.csv      노이즈별 seed-level 통계 요약
│   │   └── weight_sweep_summary.csv 큰 λ 비교 요약
│   ├── histories/           final run별 epoch history
│   ├── pilot_results.csv    noise별 λ pilot 결과
│   └── results.csv          seed별 test classification 결과
├── src/
│   ├── dataset.py           데이터 준비와 DataLoader
│   ├── model.py             shared encoder, classifier와 decoder
│   ├── experiment.py        pilot, 학습, 평가와 결과 저장
│   └── plot.py              최종 figure 생성
├── .gitignore
├── main.py                  단일 실행 진입점
├── README.md
└── requirements.txt
```

## Figure 목록

`python main.py plot`은 아래 16개 파일만 생성하고, `outputs/figures/`의 다른 PNG는
모든 figure가 정상 생성된 뒤 삭제한다.

| 파일 | 내용 |
|---|---|
| `s1_cost.png` | Baseline과 multitask의 학습·추론 비용 |
| `s2_lambda.png` | Noise별 λ pilot 결과 |
| `s3_noise_grid.png` | 숫자 0–9의 clean·세 noise 비교 |
| `s4_hparams.png` | 학습 hyperparameter |
| `s5_awgn_base.png`, `s5_awgn_multi.png` | AWGN 학습 곡선 |
| `s5_blur_base.png`, `s5_blur_multi.png` | Motion blur 학습 곡선 |
| `s5_contrast_base.png`, `s5_contrast_multi.png` | Reduced contrast + AWGN 학습 곡선 |
| `s6_recon.png` | 숫자 5·3의 noisy·reconstructed·clean 비교 |
| `s7_recon_loss.png` | Noise별 reconstruction MSE 곡선 |
| `accuracy_delta.png` | Seed별 test accuracy delta와 95% CI |
| `recall_delta.png` | 숫자 class별 recall delta |
| `accuracy_table.png` | 30-seed 최종 정확도와 paired t-test |
| `noise_example.png` | 숫자 5의 clean·세 noise 비교 |

## References

- Basu, S., Karki, M., Ganguly, S., DiBiano, R., Mukhopadhyay, S., & Nemani,
  R. (2015). [Learning Sparse Feature Representations using Probabilistic
  Quadtrees and Deep Belief Nets](https://www.esann.org/sites/default/files/proceedings/legacy/es2015-40.pdf).
  *Proceedings of the 23rd European Symposium on Artificial Neural Networks*,
  367–372. [n-MNIST dataset page](https://www.csc.lsu.edu/~saikat/n-mnist/).
- PyTorch. [Neural Networks](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html),
  *Deep Learning with PyTorch: A 60 Minute Blitz*.
