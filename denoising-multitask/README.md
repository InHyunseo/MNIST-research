# Denoising-Auxiliary Classification on n-MNIST

노이즈가 있는 MNIST 분류에서 clean image 복원을 auxiliary task로 추가했을 때
classification accuracy가 향상되는지 확인한다. 동일한 LeNet encoder와 classification
head를 사용하고 decoder 유무만 바꾼다.

## 실험 조건

- Noise: AWGN, motion blur, reduced contrast + AWGN
- Baseline: `encoder → classification head`
- Multitask: `encoder → classification head + denoising decoder`
- Loss: baseline은 cross-entropy, multitask는 `cross-entropy + λ × MSE`
- Seeds: `0, 1, 2, 3, 4`
- Optimizer: Adam, learning rate `0.001`
- Batch size: `128`, maximum epochs: `30`, validation ratio: `0.1`

Multitask는 각 noise의 seed 0에서 `λ ∈ {0.05, 0.1, 0.2}`를 비교해 validation
classification accuracy가 가장 높은 값을 선택한 뒤 모든 seed를 처음부터 학습한다.

## 설치

Python 3.10 이상 환경에서 dependency를 설치한다.

```bash
python -m pip install -r requirements.txt
```

## 데이터 준비

LSU n-MNIST archive 세 개를 `data/raw/`에 배치한다.

```text
mnist-with-awgn.gz
mnist-with-motion-blur.gz
mnist-with-reduced-contrast-and-awgn.gz
```

다음 명령은 archive를 최초 한 번만 MAT로 추출하고, n-MNIST와 순서가 일치하는
DeepLearnToolbox의 clean `mnist_uint8.mat`를 준비한다.

```bash
python main.py data
```

## 실행

Baseline과 multitask는 독립적으로 실행한다. `--device`는 `auto`, `cpu`, `cuda`를
지원하며 기본값은 `auto`다.

```bash
python main.py train-baseline --device cuda
python main.py train-multitask --device cuda
python main.py plot
```

`plot`은 학습을 실행하지 않고 저장된 final result와 history만 읽는다.

## 출력

```text
outputs/
├── checkpoints/       final run별 best-validation checkpoint
├── histories/         final run별 epoch CSV
├── figures/           accuracy 비교, delta와 history figure
├── pilot_results.csv  noise별 λ 후보의 validation 결과
└── results.csv        final seed별 test classification 결과
```

Checkpoint는 validation classification accuracy로 선택하며 test set은 최종 평가에만
사용한다. 완료된 현재 설정의 checkpoint가 있으면 다시 학습하지 않고 재사용한다.
