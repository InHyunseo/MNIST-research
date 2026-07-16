# LeNet 분류 + 복원 보조 과제 Multitask 구현 계획

## 1. 목표와 실험 범위

현재 LeNet baseline의 convolution encoder에 두 장의 `28×28` 원본 숫자를 복원하는
decoder를 연결한다. 분류와 복원을 처음부터 함께 학습해 복원 보조 과제가 겹친 숫자
Top-2 분류 성능에 주는 영향을 측정한다.

- Baseline loss: `BCEWithLogitsLoss`
- Multitask loss: `BCEWithLogitsLoss + λ_rec × balanced PIT-L1`
- Pilot: seed 0, `λ_rec ∈ {0.05, 0.1, 0.2}`
- 본 실험: 선택된 λ로 seeds 0–9를 새로 학습
- 주 평가: Overall 및 Low/Middle/High Top-2 exact-match, pair별 성능
- 기존 baseline checkpoint와 결과는 수치나 fingerprint를 바꾸지 않고 재사용
- 성능 개선은 구현 완료 조건이 아니며 개선·무효·악화를 같은 paired 분석으로 보고

LeNet encoder와 classification head는 얼리지 않는다. Decoder는 학습 때만 보조
supervision을 제공하고, 분류 추론에서는 제거할 수 있다.

## 2. 코드와 결과 구조

```text
main.py                         # 두 모델의 통합 실행·평가·시각화 CLI
mnist_overlap/
├── config.py                 # 공통 데이터·baseline 설정과 경로
├── manifest.py               # source split·manifest 생성과 overlap 좌표
├── data.py                   # 공통 Dataset·합성과 prepare_data 공개 API
├── model.py                  # checkpoint-compatible 공통 LeNet
├── metrics.py                # 공통 분류·bootstrap 지표
├── runtime.py                # seed, device, DataLoader, 원자 저장
├── main.py                   # 기존 baseline 명령 호환 wrapper
├── baseline/
│   ├── __init__.py
│   ├── main.py               # baseline 전체 실행 및 CLI
│   ├── training.py
│   ├── inference.py
│   ├── evaluation.py
│   └── plot.py
└── multitask/
    ├── __init__.py
    ├── config.py
    ├── model.py
    ├── losses.py
    ├── training.py
    ├── evaluation.py
    ├── plot.py
    └── main.py
```

- 기존 baseline 전용 실행 코드는 `baseline/`에 둔다.
- 저장소 최상위 `main.py`는 `--model baseline|multitask|all`로 파이프라인을 선택하고,
  `--skip-training`과 `--plot`을 두 모델에 동일하게 전달한다.
- 상위 `mnist_overlap/main.py`는 `baseline.main.main`을 호출하는 wrapper만 유지한다.
- 데이터·manifest·LeNet·분류 지표·runtime은 상위 공통 모듈에서 재사용한다.
- 기존 `MnistONet.layers.*` parameter key를 보존하고 `encode()`와
  `classify_features()`만 추가해 이전 checkpoint를 그대로 읽는다.
- Dataset은 기본적으로 기존 sample만 반환한다. `include_source_images=True`일 때만
  `[2,28,28]` source image를 추가해 baseline 동작과 메모리 사용을 유지한다.
- 기존 결과는 `outputs/baseline/`, `results/baseline/`에 두고 multitask 결과는
  `outputs/multitask/`, `results/multitask/`에 격리한다.
- `configs/mnist_multitask.yaml`은 baseline config 경로, pilot seed와 λ 후보만 받으며
  알 수 없는 key와 잘못된 값을 즉시 거부한다.

호환 wrapper는 다음 역할만 한다.

```python
"""기존 baseline 실행 명령을 보존하는 호환 진입점."""

from mnist_overlap.baseline.main import main

if __name__ == "__main__":
    main()
```

## 3. 모델

Shared encoder는 기존 LeNet의 convolution/pooling block이며 출력은
`[B,16,16,16]`이다. Classification head도 기존
`Flatten(4096) → 120 → 84 → 10`을 그대로 사용한다.

Decoder는 숫자가 canvas 안에서 이동한 정보를 보존할 수 있도록 convolution feature
전체를 투영한다.

```text
[B,16,16,16]
→ Flatten(4096)
→ Linear(4096→256) + ReLU
→ Linear(256→16×7×7) + ReLU
→ reshape [B,16,7,7]
→ ConvTranspose(16→8, k4, s2, p1) + ReLU
→ ConvTranspose(8→2, k4, s2, p1)
→ Sigmoid
→ [B,2,28,28]
```

같은 seed에서 baseline과 multitask의 초기 LeNet weight와 DataLoader 순서를 맞추기 위해
`set_random_seed(seed)` 직후 `MnistONet`을 먼저 만들고 decoder를 다음에 만든다.

## 4. Loss와 학습

각 예측·정답 source 쌍의 거리는 foreground와 background가 동일한 비중을 갖는
intensity-balanced L1으로 계산한다.

\[
d(p,t)=\frac12
\frac{\sum t|p-t|}{\sum t+\epsilon}
+
\frac12
\frac{\sum(1-t)|p-t|}{\sum(1-t)+\epsilon}
\]

두 decoder 출력에는 순서가 없으므로 sample마다 direct와 swapped 비용을 계산하고 더
작은 assignment를 선택한 뒤 batch 평균한다. 전체 loss는 다음과 같다.

\[
\mathcal{L}=\mathcal{L}_{classification}
+\lambda_{rec}\mathcal{L}_{reconstruction}
\]

- 모든 encoder, classification head, decoder parameter를 학습한다.
- Classification loss는 encoder와 head에, reconstruction loss는 encoder와 decoder에
  gradient를 전달한다.
- CSV에는 epoch, train/validation total·classification·reconstruction loss와
  exact-match를 기록한다.
- Early stopping과 best checkpoint는 baseline과 동일하게 validation exact-match만으로
  결정한다.
- Checkpoint에는 classifier/decoder state, seed, λ, best epoch, validation exact-match,
  config fingerprint와 `complete` 상태를 저장한다.
- 임시 파일에서 최종 경로로 교체해 checkpoint와 JSON을 원자적으로 저장한다.

Pilot은 세 λ를 seed 0으로 완전 학습한다. 가장 높은 validation exact-match를 선택하되,
최고값과 `0.001` 이내인 후보가 여러 개면 가장 작은 λ를 선택한다. 후보별 결과와 선택을
JSON으로 보존하며 본 실험 seed 0은 pilot checkpoint를 재사용하지 않고 새로 학습한다.

## 5. Paired 평가

Baseline과 multitask의 공통 seeds 0–9를 정확히 같은 test sample에서 비교하고 sample
metadata가 일치하지 않으면 평가를 중단한다.

- 모델별 Overall/Low/Middle/High exact-match와 macro-F1 평균±표본 표준편차
- `Multitask − Baseline` exact-match 차이의 seed×pair hierarchical bootstrap 95% CI
- headline으로 High overlap exact-match 차이 사용
- 각 overlap level의 45개 unordered pair accuracy와 대칭 `10×10` matrix
- High overlap 실제 pair와 예측 Top-2 pair의 row-normalized `45×45` confusion matrix
- Balanced PIT assignment로 matching한 ordinary L1, MSE, PSNR을 전체 및 overlap별 보고

전체 test reconstruction tensor를 메모리에 유지하지 않고 sample별 수치만 저장한다.
복원 예시 그림에 필요한 고정 sample만 checkpoint에서 다시 추론한다.

## 6. 시각화

다음 다섯 그림을 `results/multitask/figures/`에 생성한다.

- `training_curves.png`: 10-seed validation accuracy·total loss와 epoch별 평균선
- `overlap_comparison.png`: Low/Middle/High baseline·multitask 정확도와 seed 표준편차
- `pair_accuracy_difference.png`: 세 overlap level의 pair accuracy delta heatmap
- `pair_confusion_high.png`: High overlap baseline·multitask pair confusion matrix
- `reconstruction_examples.png`: 고정 Low/Middle/High 입력, 두 GT source와 PIT-matched 복원

## 7. 실행 인터페이스

기존 명령과 명시적 baseline 명령은 같은 실행 경로를 사용한다.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m mnist_overlap.main --device cuda

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m mnist_overlap.baseline.main --device cuda
```

Multitask 기본 실행은 데이터 준비, pilot, λ 선택, 10-seed 학습과 baseline paired 평가를
순서대로 수행한다. 다섯 그림은 baseline과 동일하게 `--plot` 단계에서 별도로 생성한다.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python main.py --model multitask --device cuda
```

- `--skip-training`: 완료된 pilot 선택과 checkpoint로 비교 평가만 수행
- `--plot`: 저장된 metrics, 학습 이력과 checkpoint로 그림만 다시 생성
- `--model baseline|multitask|all`: 한 모델 또는 두 모델의 파이프라인 선택

`python -m mnist_overlap.main`, `python -m mnist_overlap.baseline.main`,
`python -m mnist_overlap.multitask.main` 명령도 기존 호환 진입점으로 유지한다.

## 8. 검증과 완료 기준

- 이동한 기존 baseline checkpoint가 정상 로드되고 refactor 전후 logits가 동일함
- 호환 wrapper와 명시적 baseline entrypoint가 같은 실행 경로를 사용함
- Dataset 기본 key/shape가 유지되고 옵션 사용 시 source가 `[2,28,28]`임
- Decoder 출력이 `[B,2,28,28]`, 값 범위가 `[0,1]`임
- PIT loss가 출력 순서 교환에 불변이고 정답 출력 loss가 blank보다 낮음
- Reconstruction-only backward에서 encoder·decoder에만 필요한 gradient가 있음
- Classification-only backward에서 encoder·classification head에만 필요한 gradient가 있음
- 같은 seed의 baseline/multitask 초기 LeNet state가 동일함
- Pilot tie-break, config fingerprint, incomplete checkpoint 거부, 결과 경로 격리가 동작함
- 작은 CPU batch의 forward/backward/checkpoint smoke test가 통과함
- CUDA 서버에서는 deterministic 설정으로 한 epoch smoke run 후 전체 실험을 수행함

최종 산출물은 pilot 선택 JSON, final checkpoint 10개, 학습 CSV 10개, paired comparison
metrics JSON과 다섯 PNG다.
