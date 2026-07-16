"""Train the fixed three-digit CNN and save the best checkpoint."""

import csv
import random

import numpy as np
import torch

from static_sequence_core.config import CHECKPOINT_DIR, CHECKPOINT_PATH, LOGS_DIR, load_config
from static_sequence_core.dataset import create_loaders
from static_sequence_core.engine import evaluate_model, train_one_epoch
from static_sequence_core.model import StaticSequenceRecognizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    cfg = load_config()
    train_cfg = cfg["train"]
    set_seed(train_cfg["seed"])

    train_loader, test_loader = create_loaders(cfg)
    model = StaticSequenceRecognizer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = LOGS_DIR / "train_history.csv"
    best_exact_match = -1.0
    final_metrics = None

    with history_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "epoch", "train_loss", "train_digit_accuracy", "train_exact_match",
            "test_loss", "test_digit_accuracy", "test_exact_match",
        ])

        for epoch in range(1, train_cfg["fallback_epochs"] + 1):
            train_result = train_one_epoch(model, train_loader, optimizer)
            test_result = evaluate_model(model, test_loader)
            final_metrics = test_result.metrics
            writer.writerow([
                epoch,
                f"{train_result.loss:.6f}",
                f"{train_result.metrics.digit_accuracy:.6f}",
                f"{train_result.metrics.exact_match:.6f}",
                f"{test_result.loss:.6f}",
                f"{test_result.metrics.digit_accuracy:.6f}",
                f"{test_result.metrics.exact_match:.6f}",
            ])
            output.flush()
            print(
                f"epoch {epoch}/{train_cfg['fallback_epochs']} "
                f"train_loss={train_result.loss:.4f} "
                f"test_digit_acc={test_result.metrics.digit_accuracy:.4f} "
                f"test_exact={test_result.metrics.exact_match:.4f}"
            )

            if test_result.metrics.exact_match > best_exact_match:
                best_exact_match = test_result.metrics.exact_match
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"best checkpoint -> {CHECKPOINT_PATH}")

            reached_minimum_epochs = epoch >= train_cfg["epochs"]
            reached_accuracy_gate = (
                test_result.metrics.digit_accuracy >= train_cfg["min_digit_accuracy"]
                and test_result.metrics.exact_match >= train_cfg["min_exact_match"]
            )
            if reached_minimum_epochs and reached_accuracy_gate:
                break

    if final_metrics is None:
        raise RuntimeError("training produced no epoch results")
    if (
        final_metrics.digit_accuracy < train_cfg["min_digit_accuracy"]
        or final_metrics.exact_match < train_cfg["min_exact_match"]
    ):
        raise SystemExit(
            "accuracy gate not reached after fallback epochs: "
            f"digit_accuracy={final_metrics.digit_accuracy:.4f}, "
            f"exact_match={final_metrics.exact_match:.4f}"
        )
    print(f"training history -> {history_path}")


if __name__ == "__main__":
    main()
