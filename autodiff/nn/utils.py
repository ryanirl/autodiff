import numpy as np


def stable_bce(pred, actual):
    maximum = np.maximum(pred, 0)

    log = np.log(1.0 + np.exp(-np.abs(pred)))

    return np.sum(maximum - (pred * actual) + log)


def clip_stable(value, eps = 1e-6):
    return np.clip(value, eps, 1.0 - eps)


def to_logits(pred, eps = 1e-6):
    pred = np.clip(pred, eps, 1.0 - eps)

    logits = np.log(pred / (1.0 - pred))

    return logits





