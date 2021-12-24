import numpy as np


### --- Numerically Stable Utils --- ###

def stable_bce(pred, actual):
    maximum = np.maximum(pred, 0)
    HY = pred * actual
    log = np.log(1.0 + np.exp(-np.abs(pred)))

    return np.sum(maximum - HY + log)


def clip_stable(value):
    EPS = 1e-6

    return np.clip(value, EPS, 1.0 - EPS)


def to_logits(pred):
    EPS = 1e-06
    pred = np.clip(pred, EPS, 1.0 - EPS)
    logits = np.log(pred / (1.0 - pred))

    return logits





