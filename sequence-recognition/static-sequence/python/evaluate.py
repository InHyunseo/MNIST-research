"""Evaluate a checkpoint and log fixed-width prediction strings."""

import torch
from torch.utils.data import DataLoader

from static_sequence_core.bench import write_prediction_csv
from static_sequence_core.codec import digits_to_string
from static_sequence_core.config import CHECKPOINT_PATH, LOGS_DIR, load_config
from static_sequence_core.dataset import create_dataset
from static_sequence_core.metrics import compute_metrics
from static_sequence_core.model import StaticSequenceRecognizer


@torch.inference_mode()
def main() -> None:
    cfg = load_config()
    model = StaticSequenceRecognizer(cfg)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True))
    model.eval()

    loader = DataLoader(
        create_dataset(cfg, train=False),
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
    )
    predictions, labels = [], []
    for images, batch_labels in loader:
        predictions.append(model(images).argmax(dim=2))
        labels.append(batch_labels)
    prediction_tensor = torch.cat(predictions)
    label_tensor = torch.cat(labels)
    metrics = compute_metrics(prediction_tensor, label_tensor)

    print(f"digit_accuracy={metrics.digit_accuracy:.4f}")
    print(f"exact_match={metrics.exact_match:.4f}")
    sample_count = cfg["evaluation"]["sample_predictions"]
    for index in range(min(sample_count, metrics.sequence_count)):
        true_text = digits_to_string(label_tensor[index])
        pred_text = digits_to_string(prediction_tensor[index])
        print(f"[{index:04d}] true={true_text} pred={pred_text} correct={true_text == pred_text}")

    output_path = LOGS_DIR / "evaluation_predictions.csv"
    write_prediction_csv(output_path, label_tensor, prediction_tensor)
    print(f"predictions -> {output_path}")


if __name__ == "__main__":
    main()
