# MNIST Research

MNIST를 통제된 실험 환경으로 사용해 입력 구조, 겹침, 보조 과제, attention과 추론
runtime의 영향을 분리해 살펴보는 연구 프로젝트 모음이다. 각 프로젝트는 독립된 문제와
실행 환경을 가지며, 세부 실험 설계와 재현 명령은 해당 폴더의 README에 정리한다.

## Projects

| Project | Research focus | Status |
|---|---|---|
| [`overlap-multitask`](overlap-multitask/) | 겹친 두 숫자 분류에서 원본 복원 보조 과제가 LeNet 성능에 미치는 영향 | Multitask 실험 진행 중 |
| [`overlap-attention`](overlap-attention/) | Shared·class-conditional spatial attention이 강한 겹침에서 분류를 개선하는지 분석 | 3-seed 기준 완료·10-seed 후속 예정 |
| [`sequence-recognition`](sequence-recognition/) | 단일 숫자에서 고정 숫자열로의 확장과 PyTorch·ONNX·C++ 추론 비교 | 완료 |

## Research scope

### Overlap multitask

서로 다른 MNIST 숫자 두 개를 `76×76` canvas에 겹쳐 배치하고 두 class를 동시에
예측한다. Plain LeNet baseline과 두 원본 `28×28` 숫자를 복원하는 decoder를 결합한
multitask 모델을 같은 10개 학습 seed에서 비교한다.

- Low·Middle·High의 통제된 bounding-box overlap
- Top-2 exact-match와 class-pair별 오류 분석
- `BCEWithLogitsLoss + λ_rec × balanced PIT-L1`
- Seed 0 pilot을 통한 reconstruction loss 가중치 선택
- 동일 test pair의 baseline 대비 hierarchical bootstrap 비교

실행 방법과 현재 baseline 결과는
[`overlap-multitask/README.md`](overlap-multitask/README.md)를 참고한다.

### Overlap attention

동일한 controlled overlap 문제에서 plain LeNet, shared spatial attention과
class-conditional attention을 비교한다. 분류 성능뿐 아니라 attention AUPRC·IoU,
class-map selectivity와 계산 비용을 함께 측정한다.

저장된 3-seed 결과에서 class attention은 High overlap에서 shared attention보다 높았지만,
plain LeNet 대비 차이는 신뢰구간이 0을 포함했다. 따라서 attention이 모든 조건에서
일관되게 개선된다고 해석하지 않는다.

- [실행 안내](overlap-attention/MANUAL.md)
- [결과 요약](overlap-attention/results/summary.md)

### Sequence recognition

단일 MNIST 분류를 고정된 3자리 숫자열 인식으로 확장하고, 같은 가중치를 PyTorch eager,
Python ONNX Runtime과 C++ ONNX Runtime에서 실행해 prediction fidelity와 latency를
비교한다.

- Single digit: 5-seed CNN과 backend latency benchmark
- Fixed sequence: `32×96` 입력, 세 position head와 exact-match 평가
- PyTorch·Python ONNX·C++ ONNX prediction 일치 검증
- Variable-length sequence와 ROS2 연동은 후속 연구 범위

단계별 결과와 실행 방법은
[`sequence-recognition/README.md`](sequence-recognition/README.md)를 참고한다.

## Repository layout

```text
MNIST-research/
├── overlap-multitask/       LeNet classification + reconstruction
├── overlap-attention/       Spatial attention comparison
├── sequence-recognition/    Sequence modeling and inference benchmark
└── README.md
```

세 프로젝트는 코드를 공유하는 단일 Python package가 아니다. Config, 가상환경,
checkpoint와 실행 명령은 각 폴더에서 독립적으로 관리한다.

## Getting started

실행할 프로젝트 폴더로 이동한 뒤 해당 README의 환경 설정을 따른다.

```bash
# Classification + reconstruction
cd overlap-multitask

# Spatial attention
cd overlap-attention

# Sequence and inference benchmark
cd sequence-recognition
```

Overlap multitask의 대표 실행 명령은 다음과 같다.

```bash
cd overlap-multitask
conda activate hyunseo-mnist

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python main.py --model all --device cuda
```

## Reproducibility

- 학습 seed, data split과 평가 조건은 프로젝트별 config에 기록한다.
- Checkpoint 선택에는 validation set만 사용하고 test set은 최종 평가에 사용한다.
- 생성 데이터, checkpoint, ONNX model, log와 figure는 각 프로젝트의 `.gitignore` 정책을
  따른다.
- 결과를 인용하거나 비교할 때는 각 프로젝트에 기록된 seed 수와 실행 환경을 함께 본다.

MNIST는 실제 OCR 성능 경쟁보다 모델과 runtime의 차이를 통제해 이해하기 위한 데이터로
사용한다. 결과를 실제 객체 가림, 일반 OCR 또는 다른 데이터셋에 직접 일반화하지 않는다.
