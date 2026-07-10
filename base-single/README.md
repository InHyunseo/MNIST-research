# Base Single — MNIST Inference Benchmark

동일한 MNIST CNN을 세 가지 방식으로 서빙하고 추론 latency를 비교한다.

- **PyTorch (Python)** — eager 실행
- **ONNX Runtime (Python)**
- **ONNX Runtime (C++)**

세 방식 모두 같은 학습 가중치와 같은 전처리(normalization은 모델 graph에 포함)를 쓴다. 따라서 예측은 동일해야 하며, latency 차이는 runtime(PyTorch vs ONNX Runtime)과 언어/바인딩(Python vs C++)에서 온다.

## 구조

```text
configs/cnn.yaml            하이퍼파라미터 (모델 arch, 학습, 벤치마크)
python/
  mnist_core/               재사용 모듈: config, model, dataset, metrics, bench
  train.py                  학습 -> models/checkpoints/
  export_onnx.py            ONNX export + 검증 -> models/onnx/
  benchmark_pytorch.py      PyTorch 추론 벤치마크
  benchmark_onnx.py         ONNX Runtime(Python) 추론 벤치마크 + 8종 ablation
  visualize.py              baseline / ablation / backend comparison 결과 생성
cpp/
  include/onnx_infer.hpp    ONNX Runtime 추론 엔진 (재사용 가능한 클래스)
  include/csv_logger.hpp
  src/main.cpp              ONNX Runtime(C++) 추론 벤치마크 + 8종 ablation
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

위 명령은 저장소 루트에서 실행한다. 가상환경과 `requirements.txt`, `third_party/`는
모든 stage가 루트에서 공유한다.

ONNX Runtime C++ 라이브러리:

```bash
cd third_party
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
tar xzf onnxruntime-linux-x64-1.23.2.tgz
mv onnxruntime-linux-x64-1.23.2 onnxruntime
cd ..
```

## 실행

```bash
bash base-single/scripts/run_train.sh                 # 5 seed 학습
bash base-single/scripts/run_export.sh                # ONNX export
bash base-single/scripts/run_benchmark.sh baseline    # 3 backend x 5 seed -> logs/
bash base-single/scripts/run_visualize.sh baseline    # 그래프·표 -> results/
```

하이퍼파라미터(모델 채널 수, epoch, seed 목록, warmup/N, thread 수)는 `configs/cnn.yaml`에서 관리한다.
baseline ONNX Runtime 실행은 graph optimization과 CPU memory arena/pattern을 명시적으로 끄고, Python에서는 named output을 요청하지 않는다. 따라서 이전 커밋의 ORT default baseline과는 다른 no-optimization 기준점이다.

## ONNX Runtime Tuning Ablation

Python ONNX와 C++ ONNX에 같은 조건의 ONNX Runtime 설정 조합을 적용해 8종 비교를 만든다.

```bash
bash base-single/scripts/run_benchmark.sh tuning-ablation
bash base-single/scripts/run_visualize.sh tuning-ablation
```

생성 위치:

- `logs/tuning_ablation/`
- `results/tuning_ablation/figures/`
- `results/tuning_ablation/tables/`

비교하는 설정:

- `none` — baseline과 같은 no-optimization 기준점
- `graph` — graph fusion/constant folding 등 런타임 그래프 최적화
- `named` — Python에서 필요한 출력(`logits`)만 요청해 output 처리 오버헤드 축소
- `memory` — CPU memory arena/memory pattern으로 반복 추론 allocator 오버헤드 축소
- `graph_named`, `graph_memory`, `named_memory`, `all` — 위 설정 조합

C++ ONNX Runtime은 API상 출력 이름을 명시해 실행하므로, `named`는 Python 쪽에서는 실제 ablation이고 C++ 쪽에서는 기록용/no-op 조건에 가깝다. IO binding, output preallocation, CPU affinity 같은 특수 최적화는 공정 비교 설명을 흐릴 수 있어 제외한다.

## Backend Comparison

최종 발표용 비교는 PyTorch eager, Python ONNX, C++ ONNX 세 줄로 정리한다. ONNX 두 줄은 `tuning-ablation`의 `all` variant 로그를 재사용하고, PyTorch eager만 같은 seed/thread 조건으로 추가 측정한다.

```bash
bash base-single/scripts/run_benchmark.sh backend-comparison
bash base-single/scripts/run_visualize.sh backend-comparison
```

생성 위치:

- `logs/backend_comparison/`
- `results/backend_comparison/tables/backend_comparison_by_thread.csv`
- `results/backend_comparison/tables/backend_comparison_summary.md`
- `results/backend_comparison/tables/backend_comparison_fidelity.csv`

## 결과

- `results/tables/benchmark_summary.csv` — baseline backend별 latency 요약(mean/median/p95/throughput)
- `results/figures/` — baseline latency boxplot, throughput, confusion matrix, per-class accuracy/F1, 학습곡선
- `results/samples/` — baseline 맞은/틀린 예측 예시
- `results/tuning_ablation/` — ONNX Runtime 설정 8종의 Python ONNX vs C++ ONNX 비교
- `results/backend_comparison/` — PyTorch eager vs Python ONNX vs C++ ONNX 최종 비교 표

기본 benchmark 측정 조건: batch=1, warmup 후 N회, 추론 호출 구간만 측정. seed 5개로 반복하고, tuning/backend comparison은 intra-op thread `1`, `2`, `4`를 같은 조건으로 sweep한다.
