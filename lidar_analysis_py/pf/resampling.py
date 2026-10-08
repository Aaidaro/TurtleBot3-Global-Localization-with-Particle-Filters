import numpy as np


def normalize_log_weights(log_weights):
    m = np.max(log_weights)
    weights = np.exp(log_weights - m)
    total = np.sum(weights)

    if not np.isfinite(total) or total <= 0.0:
        return np.ones_like(weights) / float(len(weights))

    return weights / total


def effective_sample_size(weights):
    return 1.0 / np.sum(np.square(weights))


def systematic_resample(weights, rng):
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n

    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0

    return np.searchsorted(cumulative_sum, positions)
