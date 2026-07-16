# 결과 디렉터리

이 디렉터리는 학습된 checkpoint를 평가한 뒤 생성되는 최종 결과물을 보관한다.
소스 코드 설치나 manifest 검증만으로는 결과 파일이 생성되지 않는다.

```text
results/
├── figures/       Overlap 설정, 분류 성능, attention behavior
├── tables/        모델 비교, hierarchical interval, class, seed, model cost
└── summary.md     핵심 설정과 결과 요약
```

학습·평가부터 최종 결과 생성까지 한 번에 실행하려면 다음 명령을 사용한다.

```bash
bash scripts/run_all.sh --overwrite
```

계산과 figure 생성을 분리하려면 `python -m mnist_overlap experiment` 실행 후
`python -m mnist_overlap report`를 실행한다. `experiment`는 `outputs/`를,
`report`는 이 디렉터리를 채운다. 실제 학습을 실행하지 않은 checkout에서는 하위
디렉터리가 비어 있는 것이 정상이다.

45개 High-overlap class-pair 값은 해석이 모호한 heatmap 대신
`outputs/metrics/class_pair_accuracy_high.csv` 진단 log로만 유지한다. Raw attention map은
저장하지 않으며 sample별 수치 cache에서 통계와 figure를 다시 만들 수 있다.
