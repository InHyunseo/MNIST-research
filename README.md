# MNIST Sequence Recognition

MNIST 단일 숫자 분류에서 출발해 고정 길이와 가변 길이 숫자열 인식으로 확장하는
단계별 프로젝트다. 각 stage는 모델·데이터·학습·평가 코드를 독립적으로 유지하고,
안정된 실험 인프라만 저장소 루트의 `shared/`에서 공유할 예정이다.

## Repository layout

```text
base-single/       단일 숫자 CNN과 PyTorch/ONNX Runtime 추론 기준선
static-sequence/   고정 길이 숫자열 인식 (예정)
dynamic-sequence/  가변 길이 attention 기반 숫자열 인식 (예정)
shared/            stage 공통 실험 인프라 (예정)
ros2_ws/           ROS2 추론 wrapper (예정)
third_party/       루트에서 공유하는 외부 라이브러리 (git 제외)
```

현재 구현된 파이프라인은 [`base-single/`](base-single/README.md)에 있다.

## Base single quick start

저장소 루트에서 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

bash base-single/scripts/run_train.sh
bash base-single/scripts/run_export.sh
bash base-single/scripts/run_benchmark.sh baseline
bash base-single/scripts/run_visualize.sh baseline
```

ONNX Runtime C++ 설치와 세부 실험 모드는
[`base-single/README.md`](base-single/README.md)를 참고한다.
