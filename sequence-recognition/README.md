# MNIST Sequence Recognition and Inference Benchmark

이 프로젝트는 단일 MNIST 숫자 분류에서 출발해 숫자열 인식으로 문제를 확장하고,
동일한 모델을 Python과 C++ 추론 환경에 배포했을 때의 정확성과 지연 시간을 함께
검증한다. 데이터 자체의 복잡성은 낮게 유지하면서 다음 두 변화를 분리해 관찰하는 것이
핵심이다.

1. 단일 class 분류가 고정 길이 sequence 인식으로 확장될 때 입력·출력 계약과 평가
   지표가 어떻게 달라지는가?
2. 동일한 가중치와 입력을 PyTorch eager, Python ONNX Runtime, C++ ONNX Runtime에서
   실행할 때 예측 일치도와 추론 latency는 어떻게 달라지는가?

MNIST는 실제 OCR 성능 경쟁을 위한 대상이라기보다 모델 구조, sequence decoding,
runtime 차이를 이해하기 위한 통제된 실험 환경으로 사용한다.

## 연구 범위

프로젝트는 난이도를 단계적으로 높이는 구조를 따른다.

| 단계 | 문제 | 상태 |
|---|---|---|
| `base-single` | 한 장의 MNIST 숫자를 0–9 중 하나로 분류 | 완료 |
| `static-sequence` | 고정된 세 슬롯의 숫자를 세 자리 문자열로 인식 | 완료 |
| Variable-length sequence | 길이가 다른 숫자열을 읽고 종료 시점을 스스로 결정 | 후속 연구 |
| ROS2 integration | 검증된 추론 core를 node로 감싸 입력·추론·평가 pipeline 구성 | 후속 연구 |

현재 저장소는 단일 숫자와 고정 3자리 숫자열을 재현 가능한 기준선으로 제공한다.
가변 길이 모델과 ROS2 연동은 현재 결과의 일부가 아니며, 다음 연구 단계로 남겨둔다.

## 실험 설계

### 동일 계산의 비교

Backend 비교에서는 모델 외의 차이가 결과에 섞이지 않도록 다음 조건을 유지한다.

- 하나의 PyTorch checkpoint에서 ONNX 모델을 export한다.
- `uint8` 입력의 정규화를 model graph 안에 포함한다.
- Python과 C++이 동일한 raw test bytes를 읽는다.
- Warmup 이후 batch 1의 model 호출 구간만 측정한다.
- Latency를 비교하기 전에 backend 간 prediction fidelity를 확인한다.

따라서 PyTorch와 ONNX의 차이는 주로 runtime 변화로, Python ONNX와 C++ ONNX의 차이는
언어·binding 및 호출 overhead 변화로 해석할 수 있다.

### 단계별 평가

단일 숫자 단계에서는 classification accuracy, class별 성능과 latency를 측정한다. 고정
숫자열 단계에서는 각 위치의 digit accuracy뿐 아니라 세 자리를 모두 맞힌 exact-match를
사용한다. 문자열 변환과 CSV I/O는 timed inference 구간에서 제외하며, `007`과 같은
leading zero를 보존한다.

## 구현된 단계

### Single-digit baseline

`base-single`은 소형 CNN을 다섯 training seed로 학습하고 세 backend에서 실행한다.
ONNX Runtime의 graph optimization, named output, memory reuse 조합도 별도 ablation으로
측정한다.

최종 공통 ONNX Runtime 설정을 사용한 기록에서는 평균 latency가 다음과 같았다.

| Backend | Mean latency | PyTorch 대비 |
|---|---:|---:|
| PyTorch eager | 162.17 ± 17.15 µs | 1.00× |
| Python ONNX | 49.97 ± 8.12 µs | 3.25× |
| C++ ONNX | 49.26 ± 9.19 µs | 3.29× |

이 결과에서는 ONNX Runtime으로의 전환이 큰 차이를 만들었지만, 같은 ONNX Runtime을
사용한 Python과 C++ 사이의 평균 차이는 0.70 µs로 작았다. 즉 관측된 개선의 대부분은
언어 교체보다 runtime 교체에서 왔다. 자세한 구성과 결과 파일은
[`base-single/README.md`](base-single/README.md)에 정리되어 있다.

### Fixed three-digit sequence

`static-sequence`는 세 개의 `32×32` 슬롯을 이어 붙인 `32×96` 입력을 하나의 CNN으로
처리한다. 공유 feature 뒤의 세 position classifier가 `[B, 3, 10]` logits를 출력하며,
각 위치의 argmax를 연결해 문자열을 만든다.

검증된 test 결과는 다음과 같다.

| Metric | Result |
|---|---:|
| Digit accuracy | 98.79% |
| Exact match | 96.46% |
| PyTorch / Python ONNX / C++ ONNX string fidelity | 100% |

이 단계는 sequence 출력과 leading zero 처리, ONNX export, C++ 추론까지 연결할 수 있는
단순하고 안정적인 기준선이다. 입력 위치와 문자열 길이가 고정되어 있으므로 일반적인
sequence recognition 문제를 해결한 모델로 보지는 않는다. 구현 계약과 실행 방법은
[`static-sequence/README.md`](static-sequence/README.md)에 있다.

Latency는 실행 환경, thread 설정과 시스템 부하에 영향을 받는다. 위 수치는 저장된
실험의 결과이며 다른 장비의 절대 성능을 대표하지 않는다.

## 후속 연구

원래의 확장 목표는 길이가 다른 MNIST 숫자열을 하나의 이미지에서 읽고 EOS token까지
생성하는 모델이다. 고정된 세 position head를 다음 구조로 대체하는 방향을 고려한다.

```text
variable-length digit image
  → CNN spatial encoder
  → autoregressive attention decoder
  → digit tokens
  → EOS
```

후속 실험에서는 다음 항목을 먼저 명확히 해야 한다.

- 최대 sequence 길이와 fixed-canvas 또는 variable-width 입력 방식
- Digit, BOS, EOS, PAD vocabulary와 padding mask
- Teacher forcing을 사용하는 학습과 greedy decoding 종료 조건
- Exact-match, character accuracy, length accuracy, edit distance
- Attention이 실제 digit 위치를 순서대로 선택하는지에 대한 시각화와 정량 분석
- 전체 decoding loop 또는 one-step decoder의 ONNX export 가능성

Python 모델의 정확성과 decoding 동작을 먼저 검증한 뒤 ONNX와 C++ 추론을 추가한다.
ROS2가 필요해지면 standalone core를 복사하지 않고 얇은 inference node에서 호출하며,
dataset player와 evaluation node를 별도 package로 구성하는 방식을 고려한다.

## Repository layout

```text
base-single/          단일 숫자 분류와 backend·tuning benchmark
static-sequence/      고정 3자리 숫자열 학습·평가·ONNX·C++ 추론
third_party/          ONNX Runtime C++ library
requirements.txt     공통 Python dependency
```

각 stage는 config, Python model, C++ inference, script와 생성 artifact 경로를 독립적으로
관리한다. 가상환경과 ONNX Runtime C++ library만 저장소 루트에서 공유한다.

## Reproduction

기준 환경은 Python 3.10, PyTorch 2.4.1, torchvision 0.19.1과 ONNX Runtime
1.23.2 CPU backend다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

단일 숫자 benchmark와 고정 숫자열 pipeline은 해당 stage 문서의 명령을 따른다.

- [Single-digit benchmark](base-single/README.md)
- [Fixed three-digit sequence](static-sequence/README.md)

Dataset, checkpoint, ONNX model, benchmark log, figure와 C++ build output은 실행 과정에서
재생성되며 Git 추적 대상에 포함하지 않는다.
