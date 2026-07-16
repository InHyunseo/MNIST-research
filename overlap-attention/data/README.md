# 데이터 디렉터리

이 디렉터리는 원본 MNIST와 재현 가능한 Controlled Overlap MNIST manifest를 보관한다.

```text
data/
├── raw/                  torchvision MNIST
└── manifests/
    ├── config.sha256     데이터 설정 fingerprint
    ├── source_split.npz  Train/validation 원본 index 분리
    ├── train.npz
    ├── validation.npz
    └── test.npz
```

합성 이미지는 저장하지 않는다. `ControlledOverlapMnistDataset`이 원본 MNIST와 manifest의
source index 및 offset을 이용해 같은 이미지를 필요할 때 재구성한다.

Manifest에는 다음 정보가 포함된다.

- 원본 MNIST index와 class
- 두 숫자의 canvas offset과 상대 변위
- Bounding-box 및 pixel overlap ratio
- Low/Middle/High overlap level
- Paired validation/test를 묶는 pair ID

Train과 validation source index는 서로 겹치지 않는다. Validation과 test의 같은
pair ID는 원본 숫자, class 순서, 중심, 이동 방향을 공유한다.

생성 파일은 Git에 포함하지 않는다. 재생성 방법은 [MANUAL.md](../MANUAL.md)를
참고한다.
