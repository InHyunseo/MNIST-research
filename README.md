# MNIST Inference Benchmark

동일한 MNIST CNN을 세 가지 방식으로 서빙하고 추론 latency를 비교한다.

- **PyTorch (Python)** — eager 실행
- **ONNX Runtime (Python)**
- **ONNX Runtime (C++)**

세 방식 모두 같은 학습 가중치와 같은 전처리(normalization은 모델 graph에 포함)를 쓴다. 따라서 예측은 완전히 동일하고, latency 차이는 runtime(PyTorch vs ONNX Runtime)과 언어(Python vs C++)에서만 온다.

## 구조

```
configs/cnn.yaml            하이퍼파라미터 (모델 arch, 학습, 벤치마크)
python/
  mnist_core/               재사용 모듈: config, model, dataset, metrics, bench
  train.py                  학습 -> models/checkpoints/
  export_onnx.py            ONNX export + 검증 -> models/onnx/
  benchmark_pytorch.py      PyTorch 추론 벤치마크
  benchmark_onnx.py         ONNX Runtime(Python) 추론 벤치마크
  visualize.py              logs/ -> results/ 그래프·표
cpp/
  include/onnx_infer.hpp    ONNX Runtime 추론 엔진 (재사용 가능한 클래스)
  include/csv_logger.hpp
  src/main.cpp              ONNX Runtime(C++) 추론 벤치마크
models/{checkpoints,onnx}/  학습 가중치 / ONNX 모델
logs/                       벤치마크 CSV (latency, 예측)
results/{figures,tables,samples}/
scripts/run_{train,export,benchmark,visualize}.sh
```

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

ONNX Runtime C++ 라이브러리:

```bash
cd third_party
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
tar xzf onnxruntime-linux-x64-1.23.2.tgz
mv onnxruntime-linux-x64-1.23.2 onnxruntime
```

## 실행

```bash
bash scripts/run_train.sh       # 5 seed 학습
bash scripts/run_export.sh      # ONNX export
bash scripts/run_benchmark.sh   # 3 backend x 5 seed -> logs/
bash scripts/run_visualize.sh   # 그래프·표 -> results/
```

하이퍼파라미터(모델 채널 수, epoch, seed 목록, warmup/N, thread 수)는 `configs/cnn.yaml`에서 관리한다.

## 결과

- `results/tables/benchmark_summary.csv` — backend별 latency 요약(mean/median/p95/throughput)
- `results/figures/` — latency boxplot, throughput, confusion matrix, per-class accuracy/F1, 학습곡선
- `results/samples/` — 맞은/틀린 예측 예시

측정 조건: batch=1, intra-op thread=1, warmup 후 N회, 추론 호출 구간만 측정. seed 5개로 반복해 분산을 함께 리포트한다.
