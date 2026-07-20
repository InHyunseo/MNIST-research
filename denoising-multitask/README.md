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

n-MNIST의 noise 강도는 배포본에 다음과 같이 고정돼 있다.

| Noise | 고정 조건 |
|---|---|
| AWGN | SNR 9.5 |
| Motion blur | 이동 거리 5px, 반시계 방향 15° |
| Reduced contrast + AWGN | Contrast 50%, SNR 12 |

코드는 noise를 새로 생성하거나 강도를 변경하지 않고 배포된 noisy image를 그대로 사용한다.

## 모델 구조

Backbone은 PyTorch 공식 Neural Networks tutorial에서 `LeNet`으로 제공하는 구현과
동일하다.

```text
Input 1×32×32
  → Conv(1→6, 5×5) → ReLU → MaxPool2
  → Conv(6→16, 5×5) → ReLU → MaxPool2
  → Flatten(400)
  → FC(400→120) → ReLU
  → FC(120→84) → ReLU
  → FC(84→10)
```

이는 1998년 논문의 activation·subsampling·출력 방식까지 문자 그대로 재현한 원본
LeNet-5가 아니라, PyTorch가 공식 tutorial에서 사용하는 현대적 LeNet 구현이다.
Multitask 모델은 두 번째 pooling의 `16×5×5` 출력을 shared bottleneck으로 사용해
동일한 classification head와 denoising decoder로 분기한다.

## 설치

Python 3.10 이상 환경에서 dependency를 설치한다.

```bash
python -m pip install -r requirements.txt
```

## 데이터 준비

[LSU 공식 n-MNIST 배포 페이지](https://www.csc.lsu.edu/~saikat/n-mnist/)에서 받은
archive 세 개를 `data/raw/`에 배치한다.

n-MNIST를 사용한 결과를 보고할 때는 배포 페이지의 안내에 따라 Basu et al. (2015)을
본문에서 인용하고 References에 전체 서지정보를 포함한다.

```text
mnist-with-awgn.gz
mnist-with-motion-blur.gz
mnist-with-reduced-contrast-and-awgn.gz
```

다음 명령은 archive를 최초 한 번만 MAT로 추출하고 clean target을 준비한다.

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
├── figures/           accuracy 비교, delta와 seed-overlaid history figure
├── pilot_results.csv  noise별 λ 후보의 validation 결과
└── results.csv        final seed별 test classification 결과
```

Checkpoint는 validation classification accuracy로 선택하며 test set은 최종 평가에만
사용한다. 완료된 현재 설정의 checkpoint가 있으면 다시 학습하지 않고 재사용한다.

## References

- Basu et al. (2015), [Learning Sparse Feature Representations using Probabilistic
  Quadtrees and Deep Belief Nets](https://repository.lsu.edu/enviro_sciences_pubs/422/),
  ESANN 2015.
- PyTorch, [Neural Networks: LeNet](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html).
