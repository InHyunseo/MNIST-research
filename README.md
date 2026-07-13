# MNIST Overlap Attention

MNIST Overlap Attention은 한 이미지에 겹쳐 배치한 MNIST 숫자 두 개를 동시에 인식하고,
겹침 강도에 따른 CNN 성능 변화와 spatial attention의 효과를 비교하는 프로젝트다.

## 연구 질문

이 프로젝트는 다음 질문을 정량적으로 확인한다.

1. 숫자의 bounding box가 강하게 겹칠수록 일반 CNN의 인식 성능이 저하되는가?
2. 하나의 spatial attention map을 공유하면 성능 저하가 완화되는가?
3. Class별 attention map은 shared attention보다 강한 겹침에 효과적인가?
4. Class attention map은 실제로 해당 숫자의 고유 획에 집중하는가?

## 데이터

서로 다른 class의 MNIST 숫자 두 개를 `76×76` canvas 중앙에 배치하고 pixel별
maximum으로 합성한다. 정답은 두 숫자 class가 1인 10차원 multi-hot vector다.

Bounding-box overlap ratio에 따라 세 조건을 사용한다.

| 조건 | 범위 | 의미 |
|---|---:|---|
| Low | `[0.15, 0.30]` | 약한 겹침 |
| Middle | `[0.45, 0.60]` | 중간 겹침 |
| High | `[0.75, 0.90]` | 강한 겹침 |

Train과 validation은 서로 다른 MNIST train 원본을 사용한다. Validation과 test의
Low/Middle/High sample은 같은 원본 숫자, class 순서, 중심, 이동 방향을 공유하고
변위 크기만 달라진다.

| Split | 규모 |
|---|---:|
| Train | 60,000 images |
| Validation | 3,330 pairs, 9,990 images |
| Test | 10,000 pairs, 30,000 images |

## 비교 모델

세 모델은 동일한 LeNet backbone과 classifier 구조를 공유한다.

| 모델 | Spatial weighting | 역할 |
|---|---|---|
| `lenet` | 없음 | 기준 모델 |
| `shared_attention` | 입력당 map 1개 | Spatial attention 자체의 효과 확인 |
| `class_attention` | Class별 map 10개 | Class-conditional weighting 효과 확인 |

모델 구현은 각각 독립 파일에 있으며 공통 layer만 `models/backbone.py`를 사용한다.
동일한 seed에서는 세 모델의 공통 layer가 동일하게 초기화된다.

## 평가

정답 class가 항상 두 개이므로 가장 큰 logit 두 개를 예측으로 선택한다.

- Overlap level별 Top-2 exact-match
- Macro-F1과 class별 precision/recall
- High-overlap class-pair accuracy
- Pair ID 단위 paired bootstrap confidence interval
- Attention AUPRC, IoU, cross-map selectivity
- Class attention map permutation
- Parameter 수와 MAC 추정값

## 프로젝트 구조

```text
configs/                    실험 파라미터
src/mnist_overlap/          설치 가능한 Python package
  configuration.py          YAML 로딩, 검증, 경로 정의
  data/
    generation.py           Manifest 생성·검증과 image 합성
    dataset.py              Manifest 기반 지연 로딩 Dataset
  models/                   Backbone과 세 비교 모델
  training/
    engine.py               Epoch, early stopping, checkpoint
    runner.py               모델·seed별 학습 실행
  evaluation/
    metrics.py              Top-2 분류 지표
    analysis.py             Prediction, attention, bootstrap, 비용 분석
    runner.py               모델·seed별 최종 평가 실행
  reporting/
    generator.py            Metric 집계와 결과 생성 순서
    plotter.py              모든 PNG 계산과 시각화 설정
  pipeline.py               전체 실행 순서
  cli.py                    단일 command interface
data/                       MNIST 원본과 합성 manifest
models/checkpoints/         Validation best checkpoint
logs/                       학습 이력, prediction, metric
results/                    최종 표, 그림, 요약
scripts/                    전체 실험과 결과 생성 shell command
```

## 시작하기

설치와 단계별 명령은 [MANUAL.md](MANUAL.md)를 참고한다.

```bash
source .venv/bin/activate
pip install -e .
```

학습·평가 로그와 최종 표·그림을 한 번에 생성하려면 다음 명령을 실행한다.

```bash
bash scripts/run_all.sh
```

계산 단계와 결과 생성을 따로 실행할 수도 있다.

```bash
bash scripts/run_experiment.sh  # 데이터 준비, 전체 모델 학습, 평가
bash scripts/run_figures.sh     # 저장된 평가 log로 표와 그림 생성
```

## 문서

- [사용 설명서](MANUAL.md)
- [연구계획 및 실험 설계](OverlapMNIST_revised_plan.md)
- [실험 결과 요약](results/summary.md)
- [데이터 디렉터리 설명](data/README.md)
- [결과 디렉터리 설명](results/README.md)

## 해석 범위

결과는 중앙 정렬된 두 MNIST 숫자를 maximum 합성한 조건에 한정된다. MultiMNIST의
state of the art, Capsule Network보다 우수함, 실제 객체 가림 문제로의 직접 일반화는
주장하지 않는다.
