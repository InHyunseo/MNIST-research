"""
저장된 n-MNIST 데이터, checkpoint와 결과에서 발표용 figure를 생성한다.

입력:
    - 준비된 n-MNIST와 clean target
    - outputs의 checkpoint, history, pilot 및 final result

출력:
    - 데이터·모델·학습·복원·분류 결과 figure와 표 PNG

주요 기능:
    1. 데이터 및 reconstruction 정성 비교
    2. 학습 history와 reconstruction loss 시각화
    3. Paired accuracy와 class recall 비교 및 발표용 표 생성
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from scipy.stats import t as student_t, ttest_rel
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import (
    NOISE_TYPES,
    PROJECT_DIRECTORY,
    NoisyMnistDataset,
    load_mnist_arrays,
)
from src.experiment import (
    BATCH_SIZE,
    CHECKPOINT_DIRECTORY,
    HISTORY_DIRECTORY,
    LEARNING_RATE,
    MAXIMUM_EPOCHS,
    PILOT_RESULT_COLUMNS,
    PILOT_RESULTS_PATH,
    RANDOM_SEEDS,
    RESULT_COLUMNS,
    RESULTS_PATH,
)
from src.model import DenoisingAuxiliaryLeNet


FIGURE_DIRECTORY = PROJECT_DIRECTORY / "outputs" / "figures"
CONDITIONS = ("classification_only", "multitask")
NOISE_LABELS = {
    "awgn": "AWGN",
    "motion_blur": "Motion blur",
    "reduced_contrast_awgn": "Reduced contrast + AWGN",
}
CONDITION_LABELS = {
    "classification_only": "Baseline",
    "multitask": "Multitask",
}
HISTORY_FIGURE_NAMES = {
    ("awgn", "classification_only"): "s5_awgn_base.png",
    ("awgn", "multitask"): "s5_awgn_multi.png",
    ("motion_blur", "classification_only"): "s5_blur_base.png",
    ("motion_blur", "multitask"): "s5_blur_multi.png",
    ("reduced_contrast_awgn", "classification_only"): "s5_contrast_base.png",
    ("reduced_contrast_awgn", "multitask"): "s5_contrast_multi.png",
}
PRESENTATION_FIGURES = {
    "accuracy_delta.png",
    "accuracy_table.png",
    "noise_example.png",
    "recall_delta.png",
    "s1_cost.png",
    "s2_lambda.png",
    "s3_noise_grid.png",
    "s4_hparams.png",
    "s6_recon.png",
    "s7_recon_loss.png",
    *HISTORY_FIGURE_NAMES.values(),
}


def create_plots() -> None:
    """저장된 데이터와 산출물만 사용해 모든 발표용 figure를 생성한다."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"최종 결과가 없습니다: {RESULTS_PATH}\n먼저 학습을 완료하세요."
        )
    results = pd.read_csv(RESULTS_PATH)
    if tuple(results.columns) != RESULT_COLUMNS:
        raise ValueError(f"results.csv schema가 올바르지 않습니다: {RESULTS_PATH}")
    _validate_complete_conditions(results)
    pilot_results = _read_pilot_results()
    histories = _load_training_histories()
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    _plot_training_inference_cost()
    _plot_lambda_pilot_results(pilot_results)
    _plot_training_hyperparameters()
    _plot_aligned_noise_grid()
    _plot_single_noise_example()
    _plot_training_histories(histories)
    _plot_reconstruction_examples()
    _plot_reconstruction_loss(histories)
    _plot_accuracy_delta(results)
    _plot_classwise_recall_delta()
    _plot_accuracy_results_table(results)
    _remove_unused_figures()
    print(f"Figure를 생성했습니다: {FIGURE_DIRECTORY}")


def _read_pilot_results() -> pd.DataFrame:
    """Noise별 MSE reconstruction weight pilot 결과를 읽는다."""
    if not PILOT_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Pilot 결과가 없습니다: {PILOT_RESULTS_PATH}")
    results = pd.read_csv(PILOT_RESULTS_PATH)
    if tuple(results.columns) != PILOT_RESULT_COLUMNS:
        raise ValueError(
            f"pilot_results.csv schema가 올바르지 않습니다: {PILOT_RESULTS_PATH}"
        )
    expected_rows = len(NOISE_TYPES) * 3
    if len(results) != expected_rows:
        raise RuntimeError(
            f"원래 MSE pilot 결과가 완전하지 않습니다: {len(results)}/{expected_rows}"
        )
    return results.copy()


def _load_training_histories() -> dict[tuple[str, str], pd.DataFrame]:
    """모든 noise·condition의 30-seed history를 한 번만 읽는다."""
    histories_by_run = {}
    for noise_type in NOISE_TYPES:
        for condition in CONDITIONS:
            history_paths = [
                HISTORY_DIRECTORY / f"{noise_type}_{condition}_seed_{random_seed}.csv"
                for random_seed in RANDOM_SEEDS
            ]
            missing_paths = [path for path in history_paths if not path.exists()]
            if missing_paths:
                raise FileNotFoundError(
                    f"최종 학습 history가 없습니다: {missing_paths[0]}"
                )
            histories = []
            for history_path in history_paths:
                history = pd.read_csv(history_path)
                history["random_seed"] = int(history_path.stem.rsplit("_seed_", 1)[1])
                histories.append(history)
            histories_by_run[(noise_type, condition)] = pd.concat(
                histories,
                ignore_index=True,
            )
    return histories_by_run


def _validate_complete_conditions(results: pd.DataFrame) -> None:
    """각 noise·condition에 30개 seed 결과가 모두 존재하는지 확인한다."""
    expected_seeds = set(RANDOM_SEEDS)
    for noise_type in NOISE_TYPES:
        for condition in CONDITIONS:
            condition_results = results.loc[
                (results["noise_type"] == noise_type)
                & (results["condition"] == condition)
            ]
            available_seeds = set(condition_results["random_seed"].astype(int))
            if available_seeds != expected_seeds or len(condition_results) != len(
                RANDOM_SEEDS
            ):
                missing_seeds = sorted(expected_seeds - available_seeds)
                unexpected_seeds = sorted(available_seeds - expected_seeds)
                raise RuntimeError(
                    f"완료되지 않은 결과입니다: noise={noise_type}, "
                    f"condition={condition}, missing={missing_seeds}, "
                    f"unexpected={unexpected_seeds}"
                )


def _save_dark_table(
    output_name: str,
    title: str,
    columns: tuple[str, ...],
    rows: list[list[str]],
    figure_size: tuple[float, float],
    table_bbox: tuple[float, float, float, float],
    subtitle: str | None = None,
    notes: tuple[str, ...] = (),
) -> None:
    """발표용 dark-theme 표를 공통 형식으로 저장한다."""
    background = "#1F1F1F"
    figure, axis = plt.subplots(figsize=figure_size)
    figure.patch.set_facecolor(background)
    axis.set_facecolor(background)
    axis.axis("off")
    axis.text(
        0.02,
        0.95,
        title,
        transform=axis.transAxes,
        color="white",
        fontsize=15,
        va="top",
    )
    if subtitle is not None:
        axis.text(
            0.02,
            0.84,
            subtitle,
            transform=axis.transAxes,
            color="#BDBDBD",
            fontsize=9,
            va="top",
        )
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        bbox=table_bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#383838")
        cell.set_linewidth(0.7)
        cell.set_facecolor("#262626" if row_index == 0 else background)
        cell.get_text().set_color("#D8D8D8")
        if row_index == 0:
            cell.get_text().set_weight("bold")
    note_y = 0.22
    for note in notes:
        axis.text(
            0.04,
            note_y,
            f"• {note}",
            transform=axis.transAxes,
            color="#D0D0D0",
            fontsize=9,
            va="top",
        )
        note_y -= 0.07
    figure.savefig(
        FIGURE_DIRECTORY / output_name,
        dpi=180,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_training_inference_cost() -> None:
    """Baseline과 multitask의 parameter 및 forward MAC 표를 만든다."""
    baseline = DenoisingAuxiliaryLeNet(use_decoder=False)
    multitask = DenoisingAuxiliaryLeNet(use_decoder=True)
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    multitask_parameters = sum(
        parameter.numel() for parameter in multitask.parameters()
    )
    decoder_parameters = multitask_parameters - baseline_parameters
    baseline_macs = 416_520
    multitask_macs = 545_256
    parameter_increase = decoder_parameters / baseline_parameters * 100.0
    mac_increase = (multitask_macs - baseline_macs) / baseline_macs * 100.0
    rows = [
        [
            "Baseline",
            f"{baseline_parameters:,}",
            f"{baseline_parameters:,}",
            f"{baseline_macs:,}",
            "No",
        ],
        [
            "Multitask training",
            f"{multitask_parameters:,}",
            f"{multitask_parameters:,}",
            f"{multitask_macs:,}",
            "Used",
        ],
        [
            "Multitask inference",
            f"{multitask_parameters:,}",
            f"{baseline_parameters:,}",
            f"{baseline_macs:,}",
            "Skipped",
        ],
    ]
    _save_dark_table(
        "s1_cost.png",
        "Training vs. Inference Cost",
        (
            "Configuration",
            "Stored\nparameters",
            "Active\nparameters",
            "Forward MACs",
            "Decoder",
        ),
        rows,
        (9.2, 4.8),
        (0.02, 0.34, 0.96, 0.45),
        notes=(
            f"Decoder parameters: {decoder_parameters:,}",
            f"Stored parameters during training: +{parameter_increase:.2f}%",
            f"Forward MACs during training: +{mac_increase:.2f}%",
            "Inference MAC increase: 0%",
        ),
    )


def _plot_lambda_pilot_results(pilot_results: pd.DataFrame) -> None:
    """Noise별 lambda pilot validation accuracy와 선택값 표를 만든다."""
    rows = []
    for noise_type in NOISE_TYPES:
        noise_results = pilot_results.loc[
            pilot_results["noise_type"] == noise_type
        ].sort_values("reconstruction_weight")
        accuracy_by_weight = {
            float(row.reconstruction_weight): float(row.best_validation_accuracy)
            for row in noise_results.itertuples(index=False)
        }
        selected_mask = noise_results["selected"].map(
            lambda value: str(value).strip().lower() == "true"
        )
        selected = noise_results.loc[selected_mask]
        if len(selected) != 1:
            raise RuntimeError(f"선택된 lambda가 하나가 아닙니다: {noise_type}")
        rows.append(
            [
                NOISE_LABELS[noise_type].replace(" + ", "\n+ "),
                f"{accuracy_by_weight[0.05] * 100.0:.3f}%",
                f"{accuracy_by_weight[0.1] * 100.0:.3f}%",
                f"{accuracy_by_weight[0.2] * 100.0:.3f}%",
                f"{float(selected.iloc[0]['reconstruction_weight']):.2f}",
            ]
        )
    _save_dark_table(
        "s2_lambda.png",
        "λ Pilot Results",
        ("Noise", "λ=0.05", "λ=0.10", "λ=0.20", "Selected λ"),
        rows,
        (8.8, 3.7),
        (0.02, 0.15, 0.96, 0.55),
        subtitle="Seed 0 · Best validation accuracy",
    )


def _plot_training_hyperparameters() -> None:
    """실제로 사용한 학습 hyperparameter만 표로 만든다."""
    rows = [
        ["Optimizer", "Adam"],
        ["Learning rate", f"{LEARNING_RATE:g}"],
        ["Adam β₁, β₂", "0.9, 0.999"],
        ["Batch size", str(BATCH_SIZE)],
        ["Epochs", str(MAXIMUM_EPOCHS)],
    ]
    _save_dark_table(
        "s4_hparams.png",
        "Training Hyperparameters",
        ("Hyperparameter", "Value"),
        rows,
        (5.0, 3.8),
        (0.04, 0.08, 0.92, 0.72),
    )


def _first_test_indices(labels: tuple[int, ...]) -> np.ndarray:
    """요청 label별 첫 test index를 같은 순서로 반환한다."""
    reference = load_mnist_arrays(NOISE_TYPES[0])
    indices = []
    for label in labels:
        matches = np.flatnonzero(reference.test_labels == label)
        if matches.size == 0:
            raise RuntimeError(f"Test set에 label {label}이 없습니다.")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def _load_test_samples(
    noise_type: str, indices: np.ndarray
) -> list[dict[str, torch.Tensor]]:
    """한 noise의 지정 test sample을 Dataset과 같은 전처리로 읽는다."""
    arrays = load_mnist_arrays(noise_type)
    dataset = NoisyMnistDataset(
        arrays.test_noisy_images,
        arrays.test_clean_images,
        arrays.test_labels,
        indices,
    )
    return [dataset[index] for index in range(len(indices))]


def _format_image_axis(axis: plt.Axes) -> None:
    """Image panel의 tick과 frame을 제거한다."""
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_aligned_noise_grid() -> None:
    """같은 0–9 test sample의 clean과 세 noise version을 4×10으로 그린다."""
    labels = tuple(range(10))
    indices = _first_test_indices(labels)
    samples_by_noise = {
        noise_type: _load_test_samples(noise_type, indices)
        for noise_type in NOISE_TYPES
    }
    clean_samples = samples_by_noise[NOISE_TYPES[0]]
    row_labels = (
        "Clean",
        "AWGN",
        "Motion blur",
        "Reduced contrast\n+ AWGN",
    )
    figure, axes = plt.subplots(4, 10, figsize=(11.5, 4.6))
    for column, label in enumerate(labels):
        axes[0, column].imshow(
            clean_samples[column]["clean_target"].squeeze().numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[0, column].set_title(str(label), fontsize=9)
        for row, noise_type in enumerate(NOISE_TYPES, start=1):
            axes[row, column].imshow(
                samples_by_noise[noise_type][column]["noisy_image"].squeeze().numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
    for row in range(4):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=9, labelpad=8)
        for column in range(10):
            _format_image_axis(axes[row, column])
    figure.subplots_adjust(
        left=0.09,
        right=0.995,
        bottom=0.02,
        top=0.93,
        wspace=0.07,
        hspace=0.07,
    )
    figure.savefig(
        FIGURE_DIRECTORY / "s3_noise_grid.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_single_noise_example() -> None:
    """동일한 숫자 5의 clean과 세 noise version을 한 행으로 그린다."""
    indices = _first_test_indices((5,))
    samples_by_noise = {
        noise_type: _load_test_samples(noise_type, indices)[0]
        for noise_type in NOISE_TYPES
    }
    images = (
        samples_by_noise[NOISE_TYPES[0]]["clean_target"],
        samples_by_noise["awgn"]["noisy_image"],
        samples_by_noise["motion_blur"]["noisy_image"],
        samples_by_noise["reduced_contrast_awgn"]["noisy_image"],
    )
    titles = (
        "Clean",
        "AWGN",
        "Motion Blur",
        "Reduced Contrast + AWGN",
    )
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(
            image.squeeze().numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=14)
        _format_image_axis(axis)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.85, wspace=0.05)
    figure.savefig(
        FIGURE_DIRECTORY / "noise_example.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _load_checkpoint_model(
    noise_type: str,
    condition: str,
    random_seed: int,
    device: torch.device,
) -> DenoisingAuxiliaryLeNet:
    """지정 final checkpoint를 inference 가능한 model로 복원한다."""
    checkpoint_path = CHECKPOINT_DIRECTORY / (
        f"{noise_type}_{condition}_seed_{random_seed}.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint가 없습니다: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model = DenoisingAuxiliaryLeNet(use_decoder=condition == "multitask").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _plot_reconstruction_examples() -> None:
    """Seed 0 best checkpoint의 noisy·reconstruction·clean 예시를 그린다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_labels = (5, 3)
    indices = _first_test_indices(target_labels)
    figure, axes = plt.subplots(3, 6, figsize=(12, 5.4))
    row_labels = {
        "awgn": "AWGN",
        "motion_blur": "Motion blur",
        "reduced_contrast_awgn": "Reduced contrast\n+ AWGN",
    }
    for row, noise_type in enumerate(NOISE_TYPES):
        samples = _load_test_samples(noise_type, indices)
        noisy_images = torch.stack([sample["noisy_image"] for sample in samples]).to(
            device
        )
        clean_targets = torch.stack([sample["clean_target"] for sample in samples]).to(
            device
        )
        labels = [int(sample["label"]) for sample in samples]
        model = _load_checkpoint_model(noise_type, "multitask", 0, device)
        with torch.no_grad():
            reconstructions = model(
                noisy_images,
                include_reconstruction=True,
            )["reconstruction"]
        if reconstructions is None:
            raise RuntimeError(f"Reconstruction 출력이 없습니다: {noise_type}")

        for sample_index in range(len(indices)):
            noisy = noisy_images[sample_index]
            reconstruction = reconstructions[sample_index]
            clean = clean_targets[sample_index]
            noisy_mse = functional.mse_loss(noisy, clean).item()
            reconstruction_mse = functional.mse_loss(reconstruction, clean).item()
            images = (
                noisy,
                reconstruction.clamp(0.0, 1.0),
                clean,
            )
            titles = ("Noisy", "Reconstructed", "Clean")
            mse_labels = (
                f"MSE {noisy_mse:.4f}",
                f"MSE {reconstruction_mse:.4f}",
                "",
            )
            for image_index, (image, title, mse_label) in enumerate(
                zip(images, titles, mse_labels)
            ):
                column = sample_index * 3 + image_index
                axis = axes[row, column]
                axis.imshow(
                    image.squeeze().detach().cpu().numpy(),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                axis.set_xlabel(mse_label, fontsize=9, labelpad=3)
                if row == 0:
                    axis.set_title(
                        f"{title}\nDigit {labels[sample_index]}",
                        fontsize=10,
                    )
                _format_image_axis(axis)
        axes[row, 0].set_ylabel(row_labels[noise_type], fontsize=10, labelpad=12)
        del model

    figure.tight_layout(pad=0.8, w_pad=0.5, h_pad=1.1)
    figure.savefig(
        FIGURE_DIRECTORY / "s6_recon.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_reconstruction_loss(
    histories_by_run: dict[tuple[str, str], pd.DataFrame],
) -> None:
    """세 noise의 30-seed mean reconstruction MSE를 세로로 그린다."""
    means = {}
    all_values = []
    for noise_type in NOISE_TYPES:
        history = histories_by_run[(noise_type, "multitask")]
        required_columns = {
            "epoch",
            "training_reconstruction_loss",
            "validation_reconstruction_loss",
        }
        missing_columns = required_columns - set(history.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"Reconstruction history가 없습니다: {missing_text}")
        mean_history = history.groupby("epoch", as_index=False)[
            ["training_reconstruction_loss", "validation_reconstruction_loss"]
        ].mean()
        means[noise_type] = mean_history
        all_values.append(
            mean_history[
                ["training_reconstruction_loss", "validation_reconstruction_loss"]
            ].to_numpy(dtype=np.float64)
        )
    combined_values = np.concatenate(all_values, axis=0)
    lower_limit = max(
        0.0,
        float(np.floor((combined_values.min() - 0.001) * 1000.0) / 1000.0),
    )
    upper_limit = float(np.ceil((combined_values.max() + 0.001) * 1000.0) / 1000.0)

    figure, axes = plt.subplots(3, 1, figsize=(6.4, 8.6), sharex=True, sharey=True)
    for index, noise_type in enumerate(NOISE_TYPES):
        mean_history = means[noise_type]
        axis = axes[index]
        axis.plot(
            mean_history["epoch"],
            mean_history["training_reconstruction_loss"],
            color="#4C78A8",
            linewidth=1.8,
            label="Train",
        )
        axis.plot(
            mean_history["epoch"],
            mean_history["validation_reconstruction_loss"],
            color="#F58518",
            linewidth=1.8,
            label="Validation",
        )
        axis.set_title(NOISE_LABELS[noise_type])
        axis.set_xlim(1, MAXIMUM_EPOCHS)
        axis.set_xticks((1, 5, 10, 15, 20, 25, 30))
        axis.set_ylim(lower_limit, upper_limit)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=True)
    figure.supxlabel("Epoch")
    figure.subplots_adjust(
        left=0.11,
        right=0.98,
        bottom=0.08,
        top=0.96,
        hspace=0.28,
    )
    figure.savefig(
        FIGURE_DIRECTORY / "s7_recon_loss.png",
        dpi=180,
    )
    plt.close(figure)


def _paired_accuracies(
    results: pd.DataFrame, noise_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """한 noise의 seed-aligned baseline과 multitask test accuracy를 반환한다."""
    noise_results = results.loc[results["noise_type"] == noise_type]
    baseline = noise_results.loc[
        noise_results["condition"] == "classification_only",
        ["random_seed", "test_accuracy"],
    ].rename(columns={"test_accuracy": "baseline_accuracy"})
    multitask = noise_results.loc[
        noise_results["condition"] == "multitask",
        ["random_seed", "test_accuracy"],
    ].rename(columns={"test_accuracy": "multitask_accuracy"})
    paired = baseline.merge(multitask, on="random_seed", validate="one_to_one")
    if paired.empty:
        raise RuntimeError(f"Paired seed 결과가 없습니다: {noise_type}")
    paired = paired.sort_values("random_seed")
    return (
        paired["baseline_accuracy"].to_numpy(dtype=np.float64),
        paired["multitask_accuracy"].to_numpy(dtype=np.float64),
    )


def _confidence_interval_half_width(values: np.ndarray) -> float:
    """Sample mean의 95% t-confidence interval half width를 반환한다."""
    if len(values) <= 1:
        return 0.0
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    critical_value = float(student_t.ppf(0.975, df=len(values) - 1))
    return critical_value * standard_error


def _plot_accuracy_delta(results: pd.DataFrame) -> None:
    """같은 seed의 test accuracy 차이와 평균±95% 신뢰구간을 그린다."""
    paired_deltas = []
    mean_deltas = []
    confidence_interval_half_widths = []
    for noise_type in NOISE_TYPES:
        baseline, multitask = _paired_accuracies(results, noise_type)
        deltas = (multitask - baseline) * 100.0
        paired_deltas.append(deltas)
        mean_deltas.append(float(deltas.mean()))
        confidence_interval_half_widths.append(_confidence_interval_half_width(deltas))

    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(NOISE_TYPES))
    for noise_index, (position, deltas) in enumerate(zip(positions, paired_deltas)):
        jitter = np.linspace(-0.12, 0.12, len(deltas)) if len(deltas) > 1 else 0.0
        axis.scatter(
            position + jitter,
            deltas,
            s=34,
            color="#72A5D3",
            alpha=0.72,
            edgecolors="none",
            zorder=3,
        )
        axis.errorbar(
            position,
            mean_deltas[noise_index],
            yerr=confidence_interval_half_widths[noise_index],
            fmt="o",
            markersize=7,
            color="#173F5F",
            ecolor="#173F5F",
            elinewidth=1.7,
            capsize=5,
            label="95% Confidence Interval" if noise_index == 0 else None,
            zorder=4,
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(positions, [NOISE_LABELS[name] for name in NOISE_TYPES])
    axis.set_ylabel("Δ test accuracy")
    axis.set_ylim(-0.6, 0.6)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "accuracy_delta.png", dpi=160)
    plt.close(figure)


def _plot_classwise_recall_delta() -> None:
    """30개 paired seed의 class recall delta mean을 10×3 heatmap으로 그린다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_deltas = []
    for noise_type in NOISE_TYPES:
        print(f"Class recall을 계산합니다: noise={noise_type} device={device}")
        arrays = load_mnist_arrays(noise_type)
        noisy_images = (
            torch.from_numpy(arrays.test_noisy_images.reshape(-1, 1, 28, 28))
            .to(torch.float32)
            .div_(255.0)
        )
        noisy_images = functional.pad(noisy_images, (2, 2, 2, 2))
        labels = torch.from_numpy(arrays.test_labels).to(torch.long)
        data_loader = DataLoader(
            TensorDataset(noisy_images, labels),
            batch_size=1024,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        recalls_by_condition = {}
        for condition in CONDITIONS:
            seed_recalls = []
            for random_seed in RANDOM_SEEDS:
                model = _load_checkpoint_model(
                    noise_type,
                    condition,
                    random_seed,
                    device,
                )
                correct_by_class = torch.zeros(10, dtype=torch.float64)
                total_by_class = torch.zeros(10, dtype=torch.float64)
                with torch.no_grad():
                    for noisy_batch, label_batch in data_loader:
                        logits = model(
                            noisy_batch.to(device, non_blocking=True),
                            include_reconstruction=False,
                        )["classification_logits"]
                        predictions = logits.argmax(dim=1).cpu()
                        total_by_class += torch.bincount(
                            label_batch,
                            minlength=10,
                        ).to(torch.float64)
                        correct_by_class += torch.bincount(
                            label_batch[predictions == label_batch],
                            minlength=10,
                        ).to(torch.float64)
                seed_recalls.append((correct_by_class / total_by_class * 100.0).numpy())
                del model
            recalls_by_condition[condition] = np.stack(seed_recalls, axis=0)
        paired_delta = (
            recalls_by_condition["multitask"]
            - recalls_by_condition["classification_only"]
        )
        mean_deltas.append(paired_delta.mean(axis=0))

    values = np.stack(mean_deltas, axis=1)
    values[np.abs(values) < 0.0005] = 0.0
    limit = max(0.3, float(np.ceil(np.abs(values).max() * 10.0) / 10.0))
    figure, axis = plt.subplots(figsize=(5.8, 7.2))
    image = axis.imshow(
        values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axis.set_xticks(
        range(3),
        ("AWGN", "Motion\nblur", "Reduced contrast\n+ AWGN"),
    )
    axis.set_yticks(range(10), [str(digit) for digit in range(10)])
    axis.tick_params(axis="both", length=0)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            text_color = "white" if abs(value) >= limit * 0.5 else "#222222"
            axis.text(
                column,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.055, pad=0.04)
    colorbar.set_label("Δ recall", rotation=270, labelpad=16)
    colorbar.outline.set_linewidth(0.6)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIRECTORY / "recall_delta.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_accuracy_results_table(results: pd.DataFrame) -> None:
    """30-seed accuracy summary, confidence interval과 paired t-test 표를 만든다."""
    rows = []
    significant_rows = set()
    for row_index, noise_type in enumerate(NOISE_TYPES, start=1):
        baseline, multitask = _paired_accuracies(results, noise_type)
        baseline_percent = baseline * 100.0
        multitask_percent = multitask * 100.0
        deltas = multitask_percent - baseline_percent
        half_width = _confidence_interval_half_width(deltas)
        p_value = float(ttest_rel(multitask, baseline).pvalue)
        rows.append(
            [
                NOISE_LABELS[noise_type].replace(" + ", " +\n"),
                f"{baseline_percent.mean():.3f} ± {baseline_percent.std(ddof=1):.3f}%",
                f"{multitask_percent.mean():.3f} ± {multitask_percent.std(ddof=1):.3f}%",
                f"{deltas.mean():+.3f} pp",
                f"[{deltas.mean() - half_width:+.3f}, {deltas.mean() + half_width:+.3f}]",
                f"p = {p_value:.4f}",
            ]
        )
        if p_value < 0.05:
            significant_rows.add(row_index)

    figure, axis = plt.subplots(figsize=(11.5, 2.35))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=(
            "Noise (n = 30)",
            "Baseline",
            "Multitask",
            "Δ Accuracy",
            "95% CI",
            "Paired t-test",
        ),
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.0, 1.0, 1.0),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#2F5BD3")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor("#F7F9FF")
            cell.get_text().set_weight("bold")
        elif column_index == 0:
            cell.get_text().set_ha("left")
        if row_index in significant_rows and column_index > 0:
            cell.get_text().set_weight("bold")
    figure.savefig(
        FIGURE_DIRECTORY / "accuracy_table.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _remove_unused_figures() -> None:
    """모든 새 figure가 생성된 뒤 발표 목록 밖의 PNG만 삭제한다."""
    missing_names = sorted(
        name for name in PRESENTATION_FIGURES if not (FIGURE_DIRECTORY / name).exists()
    )
    if missing_names:
        raise RuntimeError(
            "생성되지 않은 발표용 figure가 있습니다: " + ", ".join(missing_names)
        )
    removed_names = []
    for figure_path in FIGURE_DIRECTORY.glob("*.png"):
        if figure_path.name not in PRESENTATION_FIGURES:
            figure_path.unlink()
            removed_names.append(figure_path.name)
    if removed_names:
        print(
            "사용하지 않는 figure를 삭제했습니다: " + ", ".join(sorted(removed_names))
        )


def _plot_training_histories(
    histories_by_run: dict[tuple[str, str], pd.DataFrame],
) -> None:
    """Noise와 condition별 train·validation history를 공통 축으로 저장한다."""
    maximum_loss = 0.0
    minimum_accuracy = 1.0
    for combined_history in histories_by_run.values():
        run_maximum_loss = (
            combined_history[["training_total_loss", "validation_total_loss"]]
            .to_numpy(dtype=np.float64)
            .max()
        )
        run_minimum_accuracy = (
            combined_history[["training_accuracy", "validation_accuracy"]]
            .to_numpy(dtype=np.float64)
            .min()
        )
        maximum_loss = max(maximum_loss, float(run_maximum_loss))
        minimum_accuracy = min(minimum_accuracy, float(run_minimum_accuracy))

    if not np.isfinite(maximum_loss) or maximum_loss <= 0.0:
        raise ValueError("History loss가 올바른 양수가 아닙니다.")
    if not np.isfinite(minimum_accuracy) or not 0.0 <= minimum_accuracy <= 1.0:
        raise ValueError("History accuracy가 0과 1 사이의 유한한 값이 아닙니다.")
    loss_upper_limit = max(0.1, float(np.ceil(maximum_loss * 10.5) / 10.0))
    accuracy_lower_limit = max(
        0.0,
        float(np.floor((minimum_accuracy - 0.01) * 20.0) / 20.0),
    )
    for (noise_type, condition), history in histories_by_run.items():
        _plot_spaghetti_history(
            history,
            noise_type,
            condition,
            loss_upper_limit,
            accuracy_lower_limit,
        )


def _plot_spaghetti_history(
    history: pd.DataFrame,
    noise_type: str,
    condition: str,
    loss_upper_limit: float,
    accuracy_lower_limit: float,
) -> None:
    """Loss와 accuracy에 train·validation seed 곡선과 평균을 그린다."""
    required_columns = {
        "epoch",
        "random_seed",
        "training_total_loss",
        "validation_total_loss",
        "training_accuracy",
        "validation_accuracy",
    }
    missing_columns = required_columns - set(history.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"History에 필요한 column이 없습니다: {missing_text}")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"training": "#4C78A8", "validation": "#F58518"}
    seed_count = history["random_seed"].nunique()
    individual_alpha = min(0.42, 0.95 / np.sqrt(seed_count))
    metrics = (
        (
            "Loss",
            "training_total_loss",
            "validation_total_loss",
            (0.0, loss_upper_limit),
        ),
        (
            "Accuracy",
            "training_accuracy",
            "validation_accuracy",
            (accuracy_lower_limit, 1.0),
        ),
    )
    for metric_index, (title, training_column, validation_column, limits) in enumerate(
        metrics
    ):
        axis = axes[metric_index]
        for _, seed_history in history.groupby("random_seed", sort=True):
            axis.plot(
                seed_history["epoch"],
                seed_history[training_column],
                color=colors["training"],
                linewidth=1.1,
                alpha=individual_alpha,
            )
            axis.plot(
                seed_history["epoch"],
                seed_history[validation_column],
                color=colors["validation"],
                linewidth=1.1,
                alpha=individual_alpha,
            )

        mean_history = history.groupby("epoch", as_index=False)[
            [training_column, validation_column]
        ].mean()
        axis.plot(
            mean_history["epoch"],
            mean_history[training_column],
            color=colors["training"],
            linewidth=2.0,
            label="Train",
        )
        axis.plot(
            mean_history["epoch"],
            mean_history[validation_column],
            color=colors["validation"],
            linewidth=2.0,
            label="Validation",
        )
        axis.set_title(title)
        axis.set_xlim(1, 30)
        axis.set_xticks((1, 5, 10, 15, 20, 25, 30))
        axis.set_ylim(*limits)
        axis.grid(alpha=0.25)

    axes[1].legend()
    figure.supxlabel("Epoch")
    figure.suptitle(f"{NOISE_LABELS[noise_type]} · {CONDITION_LABELS[condition]}")
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.16,
        top=0.80,
        wspace=0.12,
    )
    figure.savefig(
        FIGURE_DIRECTORY / HISTORY_FIGURE_NAMES[(noise_type, condition)],
        dpi=160,
    )
    plt.close(figure)
