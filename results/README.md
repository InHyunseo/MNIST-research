# 결과 디렉터리

이 디렉터리는 학습된 checkpoint를 평가한 뒤 생성되는 최종 결과물을 보관한다.
소스 코드 설치나 manifest 검증만으로는 결과 파일이 생성되지 않는다.

```text
results/
├── figures/       Accuracy, class-pair, attention, overlap 입력 그림
├── tables/        모델 비교, attention, bootstrap, model cost
└── summary.md     핵심 설정과 결과 요약
```

학습·평가부터 최종 결과 생성까지 한 번에 실행하려면 다음 명령을 사용한다.

```bash
bash scripts/run_all.sh
```

계산과 figure 생성을 분리하려면 `scripts/run_experiment.sh` 실행 후
`scripts/run_figures.sh`를 실행한다. `experiment`는 `models/checkpoints/`와 `logs/`를,
`report`는 이 디렉터리를 채운다. 실제 학습을 실행하지 않은 checkout에서는 하위
디렉터리가 비어 있는 것이 정상이다.
