import numpy as np

CONST = .5*np.log(2*np.pi*np.exp(1))


def zero_mean_unit_variance(data, mean=None, std=None):
    # zero mean unit variance normalization
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    return (data - mean) / std


def normalize(data, col_max):
    return data/col_max


def entropy_from_cov(cov):
    ent = cov.shape[0] * CONST + .5 * np.linalg.slogdet(cov)[1].item()
    return ent


def distance(p0, p1):
    # euclidean distance
    return np.linalg.norm(np.array(p0) - np.array(p1))