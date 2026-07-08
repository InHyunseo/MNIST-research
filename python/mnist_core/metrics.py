"""정확도 / confusion / per-class precision·recall·F1."""
import numpy as np


def accuracy(preds, trues):
    return (np.asarray(preds) == np.asarray(trues)).mean()


def confusion(true, pred):
    C = np.zeros((10, 10), dtype=int)
    np.add.at(C, (np.asarray(true), np.asarray(pred)), 1)
    return C


def per_class_prf(C):
    tp = np.diag(C).astype(float)
    recall = tp / C.sum(1)          # 클래스별 accuracy
    precision = tp / C.sum(0)
    f1 = 2 * precision * recall / (precision + recall)
    return recall, precision, f1
