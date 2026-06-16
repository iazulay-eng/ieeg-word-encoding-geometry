"""Permutation test for neural<->GPT-2 RSA significance (Notebook 2, Step 9).

Shuffles the neural RSM's word labels (rows+cols together) and recomputes the
RSA against each GPT-2 layer to build an empirical null. Cached to .npz.
"""

import os

import numpy as np
from scipy.stats import spearmanr


def layer_permutation_test(rsm_s0, rsm_s1, gpt2_rsms, n_perms=1000, metric="spearman", cache=None):
    """Empirical null by label-shuffling the neural RSM.

    Returns a dict with, per session: observed scores, null distribution
    (n_perms x n_layers), empirical p-values, and z-scores.
    """
    layers = sorted(gpt2_rsms.keys())
    n = rsm_s0.shape[0]
    triu = np.triu_indices(n, k=1)
    gvecs = np.array([gpt2_rsms[L][triu] for L in layers])   # (n_layers, n_pairs)

    def _score(vec):
        return np.array([spearmanr(vec, g)[0] for g in gvecs])

    obs0 = _score(rsm_s0[triu])
    obs1 = _score(rsm_s1[triu])

    if cache and os.path.exists(str(cache)):
        z = np.load(str(cache))
        null0, null1 = z["null_s0"], z["null_s1"]
    else:
        null0 = np.zeros((n_perms, len(layers)))
        null1 = np.zeros((n_perms, len(layers)))
        np.random.seed(0)
        for p in range(n_perms):
            sh = np.random.permutation(n)
            null0[p] = _score(rsm_s0[np.ix_(sh, sh)][triu])
            null1[p] = _score(rsm_s1[np.ix_(sh, sh)][triu])
        if cache:
            os.makedirs(os.path.dirname(str(cache)), exist_ok=True)
            np.savez(str(cache), null_s0=null0, null_s1=null1)

    return {
        "layers": layers,
        "observed_s0": obs0, "observed_s1": obs1,
        "null_s0": null0, "null_s1": null1,
        "p_s0": (null0 >= obs0[None, :]).mean(0),
        "p_s1": (null1 >= obs1[None, :]).mean(0),
        "z_s0": (obs0 - null0.mean(0)) / null0.std(0),
        "z_s1": (obs1 - null1.mean(0)) / null1.std(0),
    }
