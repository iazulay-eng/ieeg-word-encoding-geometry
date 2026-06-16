"""Nearest-neighbor serial-position decoding (Step 12).

Ported from the validated pipeline (cell 24). Each position's S1 profile is matched
to the most-correlated S0 template; correct = same position. Chance = 1/12.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr

from . import rsa


@dataclass
class Decoding:
    accuracy: float
    confusion: np.ndarray
    predictions: list


def _predict(template, test):
    """For each test row, return the index of the most-correlated template row."""
    n = template.shape[0]
    preds = []
    for t in range(n):
        corrs = [pearsonr(test[t], template[i])[0] for i in range(n)]
        preds.append(int(np.argmax(corrs)))
    return preds


def nearest_neighbor(a, b, space="rdm", fixed_indices=None, n_words=12):
    """Decode serial position by nearest template.

    space='rdm': a, b are the 12x12 RDMs (a = S0 templates, b = S1 test).
    space='raw': a, b are grand matrices (items x channels); the mean-over-lists
                 population profiles (12 x len(fixed_indices)) are used.
    Returns Decoding(accuracy [fraction], confusion 12x12, predictions).
    """
    if space == "raw":
        template = rsa.population_matrix(a, fixed_indices, n_words)
        test = rsa.population_matrix(b, fixed_indices, n_words)
    else:
        template, test = a, b

    n = template.shape[0]
    preds = _predict(template, test)
    accuracy = sum(p == t for t, p in enumerate(preds)) / n
    confusion = np.zeros((n, n))
    for t, p in enumerate(preds):
        confusion[t, p] = 1
    return Decoding(accuracy=float(accuracy), confusion=confusion, predictions=preds)
