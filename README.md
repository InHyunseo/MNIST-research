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

세 모델과 공통 backbone은 모두 `src/mnist_overlap/models.py`에 있다.
동일한 seed에서는 세 모델의 공통 layer가 동일하게 초기화된다.

## 평가

정답 class가 항상 두 개이므로 가장 큰 logit 두 개를 예측으로 선택한다.

- Overlap level별 Top-2 exact-match
- Macro-F1과 class별 precision/recall
- Seed와 Pair ID를 함께 복원추출하는 2단계 hierarchical bootstrap
- 전체 test 기준 숫자별 recall과 High-overlap class-pair 진단 log
- Model·seed별 early-stopping 안정성
- Attention AUPRC, IoU, cross-map selectivity
- Class attention map permutation
- Parameter 수와 MAC 추정값

## 프로젝트 구조

```text
configs/                    실험 파라미터 (mnist_overlap.yaml)
src/mnist_overlap/          설치 가능한 Python package
  config.py                 경로 상수, YAML → dataclass 설정, fingerprint
  data.py                   Manifest 생성·검증, image 합성, Dataset
  models.py                 공통 backbone과 세 비교 모델
  metrics.py                Top-2 분류 지표와 공용 통계 helper
  training.py               Epoch loop, early stopping, 모델·seed별 학습 실행
  analysis.py               Attention 지표, hierarchical bootstrap, 비용 분석
  evaluation.py             Run cache 관리, 최종 평가, 집계 CSV
  reporting.py              최종 표·figure·Markdown 요약 생성
  inference.py              Checkpoint 단일 이미지 추론 (배포/ROS2 wrapper 진입점)
  cli.py                    단일 command interface (전체 pipeline 포함)
data/                       MNIST 원본과 합성 manifest
outputs/                    Checkpoint, 학습 이력, run cache, 집계 metric
results/                    최종 표, 그림, 요약
scripts/run_all.sh          전체 실험과 결과 생성
```

## 시작하기

설치와 단계별 명령은 [MANUAL.md](MANUAL.md)를 참고한다.

```bash
source .venv/bin/activate
pip install -e .
```

기존 결과를 지우고 10-seed 최종 실험을 처음 실행할 때는 다음 명령을 사용한다.

```bash
bash scripts/run_all.sh --overwrite
```

중간에 종료된 뒤에는 `--overwrite` 없이 같은 명령을 실행한다. 정상 완료된 model·seed는
건너뛰며 중단된 run만 epoch 1부터 다시 시작한다.

```bash
bash scripts/run_all.sh
```

계산 단계와 결과 생성을 따로 실행할 수도 있다. 설치 후에는 `mnist-overlap` 명령도
`python -m mnist_overlap`과 동일하게 동작한다.

```bash
python -m mnist_overlap experiment  # 데이터 준비, 전체 모델 학습, 평가
python -m mnist_overlap report      # 저장된 평가 log로 표와 그림 생성
```

최종 figure는 통제된 overlap 입력, 분류 성능·효과 크기, attention behavior의 세 파일이다.
분류와 attention의 sample 단위 수치를 NPZ로 저장하므로 통계와 figure를 바꿀 때 재학습이나
checkpoint inference를 반복하지 않는다. Raw attention map은 저장하지 않는다.

## 문서

- [사용 설명서](MANUAL.md)
- [실험 결과 요약](results/summary.md)
- [데이터 디렉터리 설명](data/README.md)
- [결과 디렉터리 설명](results/README.md)

## 해석 범위

결과는 중앙 정렬된 두 MNIST 숫자를 maximum 합성한 조건에 한정된다. MultiMNIST의
state of the art, Capsule Network보다 우수함, 실제 객체 가림 문제로의 직접 일반화는
주장하지 않는다.
