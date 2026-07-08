# Three-Stage Repository Layout Plan

## Summary

- Structure should be hybrid rather than fully duplicated or fully shared.
- `base-single`, `static-sequence`, and `dynamic-sequence` should be separate experiment packages.
- Shared experiment infrastructure should live under `shared/`.
- ROS2 should stay outside standalone code under `ros2_ws/` and wrap stage cores thinly.

## Target Layout

```text
shared/
  python/ailab_bench/        # latency, csv, ORT variants, summary helpers
  cpp/include/ailab_bench/   # csv_logger, ORT tuning option helper

base-single/
  configs/
  python/base_single_core/
  cpp/
  scripts/
  data/
  results/
  README.md

static-sequence/
  configs/
  python/static_sequence_core/
  cpp/
  scripts/
  data/
  results/
  README.md

dynamic-sequence/
  configs/
  python/dynamic_sequence_core/
  cpp/
  scripts/
  data/
  results/
  README.md

ros2_ws/
  src/
    dataset_player/
    base_single_ros/
    static_sequence_ros/
    dynamic_sequence_ros/

third_party/                 # root shared, ignored
requirements.txt
README.md                    # repo overview
```

## Key Decisions

- Move the current single-digit pipeline into `base-single/` after the current benchmark/tuning work is committed.
- Do not share model-specific code across stages:
  - dataset generation/loading
  - model architecture
  - train/export scripts
  - Python/C++ inference core
  - task-specific visualization
- Share only experiment infrastructure:
  - ONNX Runtime variant definitions: `none`, `graph`, `named`, `memory`, combinations, `all`
  - latency benchmark loop
  - CSV read/write schema
  - prediction fidelity checks
  - backend comparison summary helpers
  - C++ CSV logger and ORT session option builder
- Keep visualization mostly stage-local. Single-digit classification needs confusion matrix/per-class plots, while sequence tasks need exact-match, character accuracy, and edit-distance views.
- Keep `results/` stage-local and trackable. Keep `models/`, `logs/`, generated datasets, and `cpp/build/` ignored.

## Migration Steps

1. Commit the current benchmark/tuning consolidation first.
2. Create `base-single/` and move current `configs/`, `python/`, `cpp/`, `scripts/`, `data/README.md`, `results/`, and the current README content into it.
3. Replace the root README with a short repository overview.
4. Fix paths to be stage-local:
   - Python `ROOT` should resolve to `base-single/`.
   - C++ execution should assume the stage root as current working directory.
   - CMake should reference root-level `third_party/`.
5. Add `shared/` and extract only common benchmark/tuning utilities.
6. Add `static-sequence/`:
   - fixed canvas or fixed slot multi-digit dataset
   - fixed max length output
   - string prediction
   - same Python ONNX / C++ ONNX / backend comparison pipeline
7. Add `dynamic-sequence/`:
   - CNN encoder plus autoregressive attention decoder
   - decode loop included in the exported ONNX artifact first
   - single-call inference surface returning a variable-length string
8. Add `ros2_ws/`:
   - dataset player
   - C++ inference node
   - eval node
   - optional Python inference node
   - ROS2 wrappers should import/include stage cores rather than copying core code.

## Test Plan

- After migration:
  - run `base-single` train/export/benchmark/visualize flow
  - run Python compile check
  - run C++ CMake build
- After shared extraction:
  - confirm `base-single` filenames and metrics are consistent with the pre-migration pipeline
- After static/dynamic stages:
  - run small-N benchmark
  - check Python ONNX vs C++ ONNX prediction fidelity
  - compare `none` vs `all` tuning
  - generate PyTorch / Python ONNX / C++ ONNX summary
- After ROS2:
  - run `colcon build`
  - run dataset replay -> inference -> eval smoke test

## Assumptions

- Directory names will be `base-single`, `static-sequence`, and `dynamic-sequence`.
- This structural migration should be a separate commit after the current benchmark/tuning work.
- `third_party/` must stay root-level and ignored; never copy it into stage folders.
- The final presentation should center on `dynamic-sequence`, while `base-single` and `static-sequence` remain as development and validation evidence.
