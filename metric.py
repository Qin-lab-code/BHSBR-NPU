"""NumPy evaluation metrics used by the MindSpore training loop."""

import numpy as np


def evaluate_topk(predictions, targets):
    targets = np.asarray(targets).reshape(-1, 1)
    hits = predictions == targets
    recall = float(hits.any(axis=1).mean())
    rows, ranks = np.nonzero(hits)
    if len(rows) == 0:
        return recall, 0.0, 0.0
    reciprocal = 1.0 / (ranks + 1)
    discounted = 1.0 / np.log2(ranks + 2)
    size = targets.shape[0]
    return recall, float(reciprocal.sum() / size), float(discounted.sum() / size)
