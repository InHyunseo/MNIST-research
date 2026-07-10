"""MNIST CNN classifier.

Normalization(uint8 -> float, standardize)을 forward() 안에서 수행한다. 그래서
ONNX export 시 전처리가 graph에 포함되고, 모든 runtime이 동일한 입력 처리를 쓴다.
architecture와 상수는 configs/cnn.yaml에서 읽는다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InferModule(nn.Module):
    """입력: uint8 [N,1,28,28] (0~255). 출력: logits [N,10]."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.mean = cfg["preprocess"]["mean"]
        self.std = cfg["preprocess"]["std"]
        self.conv1 = nn.Conv2d(1, m["conv1_channels"], 3)                     # 28->26
        self.conv2 = nn.Conv2d(m["conv1_channels"], m["conv2_channels"], 3)   # 13->11
        self.fc1 = nn.Linear(m["conv2_channels"] * 5 * 5, m["fc_hidden"])
        self.fc2 = nn.Linear(m["fc_hidden"], 10)
        self.dropout = nn.Dropout(m["dropout"])

    def forward(self, x):
        x = x.to(torch.float32).div(255.0)
        x = (x - self.mean) / self.std
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 26->13
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 11->5
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
