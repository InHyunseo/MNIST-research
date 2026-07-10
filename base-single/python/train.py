"""MNIST 학습 -> models/checkpoints/mnist_cnn_s{seed}.pt.

--seed 로 여러 모델(정확도 분산·latency 반복치). --curve 시 iteration 학습곡선을
logs/train_curve.csv 기록(보통 seed 0만). 하이퍼파라미터는 configs/cnn.yaml.
실행: python python/train.py --seed 0 [--curve]
"""
import argparse

import torch
import torch.nn as nn

from mnist_core.config import load_config, CKPT_DIR, LOGS_DIR
from mnist_core.model import InferModule
from mnist_core.dataset import loaders, load_test_tensors

EVAL_EVERY = 50
CURVE_SUBSET = 2000


@torch.no_grad()
def subset_acc(model, x, y):
    model.eval()
    acc = (model(x).argmax(1) == y).float().mean().item()
    model.train()
    return acc


@torch.no_grad()
def full_acc(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curve", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    t = cfg["train"]
    torch.manual_seed(args.seed)

    train_loader, test_loader = loaders(t["batch_size"])
    sub_x, sub_y = load_test_tensors()
    sub_x, sub_y = sub_x[:CURVE_SUBSET], sub_y[:CURVE_SUBSET]

    model = InferModule(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])
    criterion = nn.CrossEntropyLoss()

    step, loss_sum, loss_cnt = 0, 0.0, 0
    curve = []
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1
            loss_sum += loss.item()
            loss_cnt += 1
            if args.curve and step % EVAL_EVERY == 0:
                curve.append((step, loss_sum / loss_cnt, subset_acc(model, sub_x, sub_y)))
                loss_sum, loss_cnt = 0.0, 0
        print(f"epoch {epoch}/{t['epochs']}  test_acc={full_acc(model, test_loader):.4f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"mnist_cnn_s{args.seed}.pt"
    torch.save(model.state_dict(), path)
    print(f"saved -> {path}")

    if args.curve:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOGS_DIR / "train_curve.csv", "w") as f:
            f.write("step,train_loss,test_acc\n")
            for s, l, a in curve:
                f.write(f"{s},{l:.6f},{a:.6f}\n")
        print(f"curve -> {LOGS_DIR / 'train_curve.csv'}")


if __name__ == "__main__":
    main()
