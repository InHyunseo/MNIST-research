# 겹침 강도에 따른 다중 숫자 인식 성능 저하와 Class-Conditional Spatial Attention의 효과

**발표일**: 2026-07-27  
**환경**: WSL2 Ubuntu 22.04, CPU-only, PyTorch

---

## 목차

1. [표기 규약](#0-표기-규약)
2. [레퍼런스 문헌](#1-레퍼런스-문헌)
3. [문제 정의](#2-문제-정의)
4. [데이터셋: Controlled Overlap MNIST](#3-데이터셋-controlled-overlap-mnist)
5. [모델](#4-모델)
6. [학습 설정](#5-학습-설정)
7. [평가](#6-평가)
8. [핵심 반증 조건](#7-핵심-반증-조건)
9. [실행 계획](#8-실행-계획)
10. [범위 사다리](#9-범위-사다리)
11. [최종 주장 범위](#10-최종-주장-범위)
12. [구현 인터페이스](#11-구현-인터페이스)

---

## 0. 표기 규약

| 태그 | 의미 |
|---|---|
| 🔵 **[논문]** | 레퍼런스 논문의 절차 또는 스펙을 그대로 사용 |
| 🟡 **[변조]** | 논문 절차를 수정하여 사용하며 이유와 영향을 명시 |
| 🔴 **[자체]** | 논문에 없는 자체 설계 |

---

## 1. 레퍼런스 문헌

| ID | 문헌 | 역할 |
|---|---|---|
| **[R1]** | Sabour, Frosst, Hinton. *Dynamic Routing Between Capsules*. NIPS 2017 | MultiMNIST 데이터 생성 및 Top-2 평가 프로토콜의 기준 |
| **[R2]** | *Capsule Network is Not More Robust than Convolutional Network*. arXiv 2103.15459 | 겹친 숫자 인식에서 spatial information 보존 필요성의 참고 근거 |
| **[R3]** | LeCun et al. *Gradient-Based Learning Applied to Document Recognition*. 1998 | LeNet-5 backbone |
| **[R4]** | *Recurrent Connections Aid Occluded Object Recognition*. arXiv 1907.08831 | 생성 파라미터로 occlusion 강도를 통제한 선례 |
| **[R5]** | Mu & Gilmer. *MNIST-C*. ICML 2019 Workshop | MNIST 파생 변형을 이용한 robustness 분석의 참고 사례 |

---

## 2. 문제 정의

### 2.1 왜 단일 MNIST가 아닌가

단일 MNIST 분류는 이미 성능이 매우 높아, 작은 구조 변경으로 발생하는 차이가 학습 seed와 최적화 오차에 묻힐 가능성이 크다.

본 프로젝트는 MNIST 숫자 두 개를 하나의 이미지에 겹쳐 배치하여 다음 조건을 만든다.

- 하나의 입력에 두 개의 class가 존재
- 겹친 획에서 일부 정보가 손실
- 겹침 강도를 연속적으로 조절 가능
- 정확도 저하와 모델 개선 효과를 조건별로 분석 가능

### 2.2 연구 질문

> 1. 겹침 강도가 증가할 때 일반 CNN의 다중 숫자 인식 성능은 어떻게 저하되는가?  
> 2. Shared spatial attention은 이 저하를 완화하는가?  
> 3. Class-conditional spatial attention은 shared attention보다 효과적인가?  
> 4. Class와 attention map의 연결을 바꾸면 성능이 저하되는가?  
> 5. 각 class attention은 실제로 해당 숫자의 고유 획에 선택적으로 집중하는가?

### 2.3 핵심 가설

Class마다 서로 다른 spatial attention map을 사용하면, 동일한 위치에 겹쳐 있는 feature를 class별로 다르게 가중할 수 있다.

\[
A_c = \sigma\!\left(g_c(F)\right), \qquad
F'_c = F \odot A_c
\]

따라서 겹침이 강할수록 plain CNN이나 shared attention보다 class-conditional attention의 이점이 커질 것으로 가정한다.

---

## 3. 데이터셋: Controlled Overlap MNIST

### 3.1 기본 조건 🔵[R1]

- 서로 다른 class의 MNIST 숫자 두 개를 선택
- 두 label을 1로 갖는 10차원 multi-hot label 사용
- MNIST train image끼리 train set 생성
- MNIST test image끼리 test set 생성
- 두 숫자를 하나의 canvas에 overlay

### 3.2 고정 캔버스

모든 입력은 동일한 크기를 사용한다.

\[
X \in \mathbb{R}^{1\times 76\times 76}
\]

캔버스를 조건별로 다르게 하면 입력 크기와 feature-map 크기가 동시에 달라지므로, 모든 train/test sample에서 \(76\times76\)으로 고정한다.

### 3.3 겹침 지표

두 \(28\times28\) bounding box의 상대 변위를

\[
d=(d_x,d_y)
\]

라고 하면 bounding-box overlap ratio는 다음과 같다.

\[
r_{\mathrm{bbox}}
=
\frac{(28-|d_x|)_+(28-|d_y|)_+}{28^2}
\]

여기서 \((x)_+=\max(x,0)\)이다.

- \(r_{\mathrm{bbox}}\approx 1\): 강한 겹침
- \(r_{\mathrm{bbox}}\approx 0\): 약한 겹침

**생성 파라미터는 상대 변위 \(d\)**이고, **주 분석축은 실제 샘플별 \(r_{\mathrm{bbox}}\)**이다.

---

### 3.4 위치 교란 제거: pair center 고정 🔴[자체]

겹침이 약해질수록 숫자가 이미지 가장자리로 이동하면, 성능 차이가 겹침 때문인지 translation 때문인지 구분할 수 없다.

따라서 두 숫자의 공통 중심 \(c\)를 canvas 중앙에 고정하고 상대 변위만 변경한다.

\[
p_a = c-\frac{d}{2},
\qquad
p_b = c+\frac{d}{2}
\]

실제 pixel 좌표에서는 정수 반올림을 적용한다.

```text
동일한 원본 숫자 쌍
동일한 pair center
상대 변위 d만 변화
```

이 설계로 겹침 강도와 전체 위치 변화를 분리한다.

---

### 3.5 겹침 구간 균형화 🔴[자체]

단순히 shift 범위를 균일하게 sampling하면 특정 \(r_{\mathrm{bbox}}\) 구간에 sample이 몰릴 수 있다.

따라서 다음 세 구간을 같은 수로 생성한다.

| 구간 | \(r_{\mathrm{bbox}}\) 범위 | 의미 |
|---|---:|---|
| Low overlap | \([0.15,0.30]\) | 약한 겹침 |
| Mid overlap | \([0.45,0.60]\) | 중간 겹침 |
| High overlap | \([0.75,0.90]\) | 강한 겹침 |

각 구간에서 45개 unordered class pair를 최대한 균등하게 구성한다.

\[
\binom{10}{2}=45
\]

필요한 \(d\)는 목표 구간을 만족할 때까지 rejection sampling한다.

---

### 3.6 합성 연산자 🔴[자체]

\[
X=\max(X_a,X_b)
\]

채택 이유:

- addition 후 clipping은 겹친 영역을 포화시킬 수 있음
- averaging은 겹침 영역의 밝기를 인위적으로 낮춤
- max는 입력 범위를 유지하면서 두 획을 하나의 canvas에 합성 가능

단, max 합성은 한 숫자의 intensity가 다른 숫자보다 작을 때 해당 정보를 제거할 수 있다. 이 정보 손실이 태스크 난이도의 일부가 된다.

---

### 3.7 Paired test set 🔴[자체]

동일한 원본 숫자 쌍 \((X_a,X_b)\)를 Low/Mid/High overlap 조건으로 반복 합성한다.

\[
\left\{
X^{\mathrm{low}}_{a,b},
X^{\mathrm{mid}}_{a,b},
X^{\mathrm{high}}_{a,b}
\right\}
\]

목적:

- 숫자 쌍 자체의 난이도를 조건 간 동일하게 유지
- 겹침 변화에 따른 within-pair 성능 비교
- 모델 간 paired bootstrap 가능
- class-pair 구성 차이로 인한 교란 감소

`pair_id`는 원본 MNIST sample index 두 개로 정의한다.

추가로 각 `pair_id`에서 두 숫자가 벌어지는 8방향 중 하나를 고정한다. Low/Mid/High
조건은 같은 원본, class 순서, pair center, 이동 방향을 공유하고 변위 크기만
바꾼다. 정수 좌표 배치 후 실제 offset 차이로 (d)와 (r_{\mathrm{bbox}})를 다시
계산한다.

---

### 3.8 GT stroke mask와 고유 획 🔴[자체]

합성 전 숫자별 mask를 저장한다.

\[
M_a=\{X_a>0.5\},
\qquad
M_b=\{X_b>0.5\}
\]

두 숫자가 동시에 차지하는 영역에서는 어느 class에 집중했는지 분리하기 어렵다. 따라서 class별 **고유 획 영역**도 정의한다.

\[
E_a=M_a\setminus M_b,
\qquad
E_b=M_b\setminus M_a
\]

- \(E_a\): 숫자 \(a\)에만 존재하는 획
- \(E_b\): 숫자 \(b\)에만 존재하는 획

Attention mechanism 분석에서는 전체 mask 일치도와 고유 획 selectivity를 함께 측정한다.

---

### 3.9 Pixel overlap ratio

\[
r_{\mathrm{pix}}
=
\frac{|M_a\cap M_b|}
{\min(|M_a|,|M_b|)}
\]

\(r_{\mathrm{pix}}\)는 숫자 모양에 의존하므로 데이터 생성의 통제 변수로 사용하지 않는다.

예를 들어 같은 위치에 놓더라도 `0+1`과 `3+8`의 \(r_{\mathrm{pix}}\) 분포는 다르다. 따라서 보조 설명 변수로만 저장하고, 분석할 때 class-pair 또는 pair_id를 통제한다.

---

### 3.10 데이터 규모와 저장 필드

| 구분 | 수량 |
|---|---:|
| Train | 60,000 |
| Validation pair | 3,330 pair |
| Validation image | 9,990 image |
| Paired test pair | 10,000 pair |
| Paired test image | 30,000 image |

MNIST train 원본은 label-stratified 방식으로 50,000개 train source와 10,000개
validation source로 먼저 나눈다. 합성 sample을 나중에 나누지 않으므로 두 split
사이에 같은 MNIST 원본이 등장하지 않는다.

저장 필드:

```text
image
label
label_a
label_b
source_index_a
source_index_b
pair_id
offset_a
offset_b
d_x
d_y
r_bbox
r_pix
mask_a
mask_b
exclusive_mask_a
exclusive_mask_b
overlap_bin
```

데이터는 한 번 생성 후 `.npz` 또는 `.pt`로 캐싱한다. 모든 모델은 동일한 파일을 사용한다.

---

## 4. 모델

### 4.1 공통 backbone: Modernized LeNet-5

```text
Input 1×76×76
→ Conv(1→6, 5×5) → ReLU → MaxPool(2×2)
→ Conv(6→16, 5×5) → ReLU
→ Attention module 또는 identity
→ MaxPool(2×2)
→ Flatten(4096)
→ FC(4096→120) → ReLU
→ FC(120→84) → ReLU
→ Output
```

원 LeNet-5에서 다음을 현대적인 형태로 변경한다.

- partial connectivity 대신 full connectivity
- trainable subsampling 대신 MaxPool
- tanh 대신 ReLU
- RBF output 대신 linear logits
- softmax 대신 multi-label logits

모든 모델은 동일한 convolutional backbone을 사용한다.

---

### 4.2 M0: Plain LeNet Multi-label

```text
Backbone
→ FC(84→10)
→ logits
```

학습 loss:

\[
\mathcal L_{\mathrm{BCE}}
=
-\sum_{c=0}^{9}
\left[
y_c\log\sigma(z_c)
+
(1-y_c)\log(1-\sigma(z_c))
\right]
\]

실제 구현에서는 numerical stability를 위해 `BCEWithLogitsLoss`를 사용한다.

---

### 4.3 M1-shared: Shared Spatial Attention

하나의 attention map을 모든 class가 공유한다.

\[
A=\sigma(\operatorname{Conv}_{1\times1}(F))
\]

\[
F'=F\odot A
\]

```text
C3 feature F: 16×32×32
→ Conv1×1(16→1)
→ sigmoid
→ shared mask A
→ F ⊙ A
→ 기존 LeNet 경로
```

목적:

> Spatial attention 자체가 plain CNN보다 효과적인지 확인한다.

---

### 4.4 M1-class: Class-Conditional Spatial Attention

각 class마다 서로 다른 attention map을 생성한다.

\[
A_c=\sigma\!\left(
\operatorname{Conv}^{(c)}_{1\times1}(F)
\right),
\qquad c=0,\dots,9
\]

\[
F'_c=F\odot A_c
\]

각 class별 feature가 shared FC를 통과한 후 해당 class logit을 계산한다.

```text
F: 16×32×32
→ Conv1×1(16→10)
→ A0, A1, ..., A9

각 class c:
F'_c = F ⊙ A_c
→ MaxPool
→ Shared FC1
→ Shared FC2
→ class-specific scalar head w_c
```

설계 가설:

> Shared attention은 하나의 공간 선택만 만들 수 있지만, class-conditional attention은 같은 위치의 feature를 class별로 다르게 가중할 수 있다.

---

`A_c \equiv 1`이고 FC를 공유하는 class-null branch는 수학적으로 M0와 같은
함수이므로 별도 실험 모델에서 제외한다. M0-wide 또한 현재 프로젝트 범위에서
제외한다.

---

### 4.5 비용 비교

입력 \(76\times76\) 기준의 예상값이다. 실제 구현 후 profiler로 다시 측정한다.

| 모델 | Parameter | MACs | 핵심 차이 |
|---|---:|---:|---|
| M0 | 505,226 | 약 3.74M | Plain baseline |
| M1-shared | 505,243 | 약 3.75M | Shared attention 추가 |
| M1-class | 505,396 | 약 8.42M | 10개 class attention branch |

핵심 비교:

| 비교 | 분석 대상 |
|---|---|
| M1-shared vs M0 | Spatial attention 자체의 효과 |
| M1-class vs M1-shared | Class-conditional 구조의 추가 효과 |
| M1-class normal vs permuted map | class와 attention map 연결의 효과 |

---

## 5. 학습 설정

모든 모델에서 다음 조건을 동일하게 유지한다.

```yaml
train:
  optimizer: Adam
  learning_rate: 0.001
  batch_size: 128
  maximum_epochs: 20
  seeds: [0, 1, 2]
  early_stopping_patience: 3
  early_stopping_minimum_delta: 0.001
  monitor: validation_exact_match
```

추가 통제:

- 동일 train/validation split
- 동일 batch order seed
- 동일 initialization seed 기록
- 입력 범위 `[0, 1]`, 추가 normalization 없음
- augmentation 없음
- 동일 early-stopping rule
- validation set에서만 hyperparameter와 attention IoU threshold 결정
- test set은 최종 평가에만 사용

---

## 6. 평가

### 6.1 Top-2 Exact Match 🔵[R1]

정답 class가 항상 두 개이므로, 가장 큰 logit 두 개를 예측 class로 선택한다.

\[
\hat Y=\operatorname{Top2}(z)
\]

\[
\mathrm{ExactMatch}
=
\mathbf 1[\hat Y=Y]
\]

주 지표는 \(r_{\mathrm{bbox}}\) 구간별 exact-match accuracy이다.

---

### 6.2 보조 성능 지표

| 지표 | 목적 |
|---|---|
| Macro-F1 | class 불균형과 class별 인식 성능 |
| Per-class precision/recall | 취약한 숫자 확인 |
| Pairwise exact accuracy | 어려운 숫자 조합 확인 |
| Parameter/MACs | 계산 비용과 capacity 공개 |

모든 분류 지표는 별도 threshold 없이 Top-2 prediction으로 계산한다.

---

### 6.3 겹침 저하 곡선

각 모델의 성능을 \(r_{\mathrm{bbox}}\)에 따라 평가한다.

\[
\mathrm{Accuracy}=f(r_{\mathrm{bbox}})
\]

메인 결과표:

| \(r_{\mathrm{bbox}}\) 구간 | M0 | M1-shared | M1-class |
|---|---:|---:|---:|
| Low |  |  |  |
| Mid |  |  |  |
| High |  |  |  |

모델의 겹침 대응 효과는 다음 차이로 표현한다.

\[
\Delta_m(r)
=
\mathrm{Acc}_m(r)-\mathrm{Acc}_{M0}(r)
\]

특히 다음을 확인한다.

\[
\Delta_{\mathrm{class}}(r)
=
\mathrm{Acc}_{M1\text{-class}}(r)
-
\mathrm{Acc}_{M1\text{-shared}}(r)
\]

High overlap에서 \(\Delta_{\mathrm{class}}\)가 커진다면 class별 spatial weighting이 강한 겹침에서
추가로 유용하다는 가설을 지지한다.

---

### 6.4 Attention 정량 평가

Attention map은 \(32\times32\)이므로 GT mask를 동일한 해상도로 downsampling한다.

#### 1) 전체 획 일치도

\[
A_{y_a}\leftrightarrow M_a,
\qquad
A_{y_b}\leftrightarrow M_b
\]

- AUPRC: continuous attention score 사용
- IoU: validation set에서 정한 threshold로 이진화

#### 2) Cross-map 고유 획 selectivity

\[
S_{\mathrm{cross}}
=
\frac{1}{2}
\left[
\operatorname{mean}(A_{y_a}-A_{y_b}\mid E_a)
+
\operatorname{mean}(A_{y_b}-A_{y_a}\mid E_b)
\right]
\]

- \(S_{\mathrm{cross}}>0\): 해당 class map이 자기 고유 획에서 상대 map보다 큼
- downsampling 후 고유 획이 5 pixel 미만인 sample은 제외
- 유효 sample 비율을 함께 보고

Shared attention은 class별 map이 없으므로 cross-map selectivity 대상에서 제외한다.

#### 3) Attention map permutation

학습된 M1-class에서 추론할 때 class map을 한 칸씩 순환 교환하고 성능 저하를 측정한다.
이는 class와 attention map의 연결이 실제 예측에 사용되는지 확인하는 보조 분석이다.

---

### 6.5 통계 처리

- seed 3개 결과를 평균 \(\pm\) 표준편차로 보고
- 동일 paired test sample에서 모델 간 정확도 차이를 paired bootstrap으로 평가
- 95% confidence interval 보고
- class-pair별 결과도 함께 확인
- High overlap 모델 차이와 High–Low difference-in-differences를 primary comparison으로 사용

---

## 7. 핵심 반증 조건

| 관측 결과 | 해석 |
|---|---|
| M0가 overlap에 따라 저하되지 않음 | 현재 태스크에서 겹침이 충분한 난이도를 만들지 못함 |
| M1-shared가 M0와 동일 | Spatial attention 자체의 이점이 없음 |
| M1-class가 M1-shared와 동일 | Class별 attention map의 추가 이점이 없음 |
| M1-class가 Low overlap에서도 항상 우세 | 개선 원인이 겹침 대응만은 아닐 가능성 |
| Attention AUPRC는 높지만 selectivity가 낮음 | 숫자 영역은 찾지만 class별 분리는 하지 못함 |
| 성능은 상승하지만 attention 지표가 개선되지 않음 | 제안한 mechanism 설명이 지지되지 않음 |
| Map permutation에도 성능이 유지됨 | class와 map의 연결이 핵심 원인이 아닐 가능성 |

Negative result도 가설의 어느 부분이 성립하지 않는지 명확히 보고한다.

---

## 8. 실행 계획

### Step 0 — 데이터와 baseline 검증

1. 데이터 생성기 구현
2. pair center와 \(r_{\mathrm{bbox}}\) 구간 검증
3. class-pair 균형 확인
4. Train/validation/test source 및 paired 조건 검증

---

### Step 1 — 핵심 ablation

1. M1-shared 구현
2. M1-class 구현
3. 세 모델을 동일 seed로 3회 반복
4. exact match와 비용 비교

---

### Step 2 — 메커니즘 분석

1. Attention AUPRC/IoU
2. Cross-map 고유 획 selectivity
3. Attention map permutation
4. High-overlap class-pair accuracy heatmap
5. paired bootstrap confidence interval
6. M0의 Low–High 저하와 overlap 구간별 \(\Delta\) 분석

M0의 Low–High 차이가 작거나 confidence interval이 0을 포함하면 실험을 중단하지 않고
현재 합성 조건에서 겹침 난이도가 충분하지 않았다는 negative result로 보고한다.

---

## 9. 범위 사다리

| 단계 | 범위 |
|---|---|
| 최소 | M0 vs M1-class, overlap 3구간, seed 1 |
| 필수 | M0 + M1-shared + M1-class, seed 3 |
| 목표 | paired bootstrap, attention AUPRC/IoU/selectivity, class-pair heatmap |
| 여유 | Attention map permutation과 예시 시각화 |

---

## 10. 최종 주장 범위

본 프로젝트가 직접 주장하려는 것은 다음과 같다.

> 고정된 다중 숫자 분류 조건에서 겹침 강도가 증가할수록 plain CNN의 성능이 어떻게 저하되는지 측정하고, 동일 backbone에 class-conditional spatial attention을 추가했을 때 그 저하가 완화되는지 검증한다. 또한 shared attention, attention–stroke 정량 일치도, class-map permutation을 통해 개선 메커니즘을 이해한다.

다음은 주장하지 않는다.

- MultiMNIST 전체의 state of the art
- Capsule Network보다 우수함
- 일반 object detection 또는 real-world occlusion에 직접 일반화됨
- 더 적은 연산량으로 더 높은 성능을 달성함

---

## 11. 구현 인터페이스

코드는 설치 가능한 `mnist_overlap` package로 구성한다. 세 모델은 공통 backbone을
공유하지만 각각 독립 파일에 정의한다. CLI와 외부 application은 각 기능 package의
공개 실행 함수를 직접 호출한다.

```text
src/mnist_overlap/
├── data/
├── models/
│   ├── backbone.py
│   ├── lenet.py
│   ├── shared_attention.py
│   └── class_attention.py
├── training/
├── evaluation/
├── reporting/
├── pipeline.py
└── cli.py
```

일반 실행은 세 개의 shell script를 사용한다.

```bash
bash scripts/run_experiment.sh
bash scripts/run_figures.sh
bash scripts/run_all.sh
```

`run_experiment.sh`는 데이터 준비부터 세 모델과 전체 seed의 학습·평가까지 실행해
log를 생성한다. `run_figures.sh`는 저장된 log에서 표와 그림을 만들며,
`run_all.sh`는 두 단계를 이어서 실행한다. 개별 단계와 option을 직접 제어할 때만
내부 module CLI인 `python -m mnist_overlap`을 사용한다.

설치와 option별 사용법은 [MANUAL.md](MANUAL.md), 프로젝트 개요는
[README.md](README.md)를 참고한다.
