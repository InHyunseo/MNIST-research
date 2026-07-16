# Static Sequence — Three-Digit MNIST Recognition

고정된 세 슬롯의 MNIST 숫자열을 하나의 CNN으로 읽고 세 자리 문자열로 변환한다.
`6 → 16 → 120 → 84` 레이어 구성의 CNN과 공유 embedding 뒤에 배치한 세 개의
position classifier를 사용한다.

## Contract

```text
input   uint8 [B, 1, 32, 96]
encoder feature [B, 120, 1, 17]
output  float32 logits [B, 3, 10]
decode  argmax per position, e.g. [0, 0, 7] -> "007"
```

각 `28×28` MNIST 이미지는 `32×32` 슬롯 중앙에 배치된다. train sequence는 MNIST
train split만, test sequence는 MNIST test split만 사용하며 config seed로 재현된다.
정규화는 모델 graph 안에 포함된다.

## Layout

```text
configs/static.yaml              단일 stage 설정
python/static_sequence_core/     dataset, codec, model, metrics, benchmark core
python/{train,evaluate}.py       학습과 평가
python/{export_onnx,dump_data}.py
python/benchmark_{pytorch,onnx}.py
cpp/                             C++17 ONNX Runtime inference
tests/                           dataset, codec, model, overfit smoke tests
scripts/                         test/train/evaluate/export/benchmark entrypoints
```

## Run

저장소 루트의 `.venv`, `requirements.txt`, `third_party/onnxruntime`을 공유한다. 모든
명령은 저장소 루트에서 실행할 수 있다.

```bash
bash static-sequence/scripts/run_test.sh
bash static-sequence/scripts/run_train.sh
bash static-sequence/scripts/run_evaluate.sh
bash static-sequence/scripts/run_export.sh
bash static-sequence/scripts/run_benchmark.sh
```

빠른 benchmark smoke test:

```bash
N=20 WARMUP=5 bash static-sequence/scripts/run_benchmark.sh
```

## Metrics and latency boundary

- `digit_accuracy`: 세 위치를 합친 digit 단위 정확도
- `exact_match`: 세 자리가 전부 일치한 sequence 비율
- timed inference: model/ORT call과 position-wise argmax 포함
- timed inference 제외: 문자열 생성, print, CSV I/O

| Backend | Mean | Median | p95 |
|---|---:|---:|---:|
| PyTorch eager | 286.6 µs | 254.0 µs | 482.0 µs |
| Python ONNX | 68.9 µs | 61.5 µs | 92.4 µs |
| C++ ONNX | 63.1 µs | 58.0 µs | 78.2 µs |

Python eager, Python ONNX, C++ ONNX는 동일한 raw test bytes를 사용하며 benchmark가
끝난 뒤 전체 10,000개 문자열의 prediction fidelity를 검사한다. 시각화와 ONNX Runtime
tuning ablation은 이 stage 범위에서 제외한다.
