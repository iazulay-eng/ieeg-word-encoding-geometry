"""RSA: serial-position RDM (split-half), second-order RSA, word RSM.

Ported from the validated fast-analysis pipeline (cells 7, 12, 12a, 12b, 17, 17c).
Split-half uses a fixed seed (42) so results are reproducible and match the cache.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as _cosine_dist
from scipy.stats import pearsonr, spearmanr


def _split_half_sims(reshaped, top_idx, n_iter):
    """Return the stack of n_iter split-half 12x12 similarity matrices."""
    n_lists, n_words, _ = reshaped.shape
    data = reshaped[:, :, top_idx]
    half = n_lists // 2
    sims = []
    np.random.seed(42)
    for _ in range(n_iter):
        shuf = np.random.permutation(n_lists)
        a, b = shuf[:half], shuf[half:half * 2]
        mean_a = np.mean(data[a], axis=0)
        mean_b = np.mean(data[b], axis=0)
        sim = np.zeros((n_words, n_words))
        for pa in range(n_words):
            for pb in range(n_words):
                sim[pa, pb] = np.corrcoef(mean_a[pa], mean_b[pb])[0, 1]
        sims.append(sim)
    return np.array(sims)


def serial_position_rdm(matrix, n_channels=100, n_iter=10, select="variance", fixed_indices=None):
    """Split-half serial-position RDM.

    In : matrix (items x channels) reshaped to (lists, 12, channels); channels
         chosen by positional variance (`select='variance'`) unless `fixed_indices`.
    Out: (mean_rdm 12x12, iters_stack (n_iter,12,12), channel_indices).
    """
    total, n_ch = matrix.shape
    n_words = 12
    n_lists = total // n_words
    reshaped = matrix[:n_lists * n_words].reshape(n_lists, n_words, n_ch)

    if fixed_indices is not None:
        top_idx = fixed_indices
    else:
        pos_var = np.var(np.mean(reshaped, axis=0), axis=0)
        top_idx = np.argsort(pos_var)[::-1][:n_channels]

    sims = _split_half_sims(reshaped, top_idx, n_iter)
    return np.mean(sims, axis=0), sims, top_idx


def rdm_with_fixed_features(matrix, top_idx, n_iter=10, n_words=12):
    """Split-half RDM on pre-selected channels (used by time-resolved windows).

    Note: the 'B' half is shuf[half:] (all remaining lists), matching the
    pipeline's time-resolved RDM (slightly different from serial_position_rdm).
    """
    total, n_ch = matrix.shape
    n_lists = total // n_words
    reshaped = matrix[:n_lists * n_words].reshape(n_lists, n_words, n_ch)
    data = reshaped[:, :, top_idx]
    half = n_lists // 2
    sims = []
    np.random.seed(42)
    for _ in range(n_iter):
        shuf = np.random.permutation(n_lists)
        a, b = shuf[:half], shuf[half:]
        mean_a = np.mean(data[a], axis=0)
        mean_b = np.mean(data[b], axis=0)
        sim = np.zeros((n_words, n_words))
        for pa in range(n_words):
            for pb in range(n_words):
                sim[pa, pb] = np.corrcoef(mean_a[pa], mean_b[pb])[0, 1]
        sims.append(sim)
    return np.mean(sims, axis=0)


def population_matrix(matrix, top_idx, n_words=12):
    """Mean-over-lists population profile per position: (n_words, len(top_idx))."""
    total, n_ch = matrix.shape
    n_lists = total // n_words
    reshaped = matrix[:n_lists * n_words].reshape(n_lists, n_words, n_ch)
    return np.mean(reshaped[:, :, top_idx], axis=0)


def trialwise_rdm(matrix, fixed_indices=None, n_channels=100, n_words=12):
    """Trial-wise RDM: correlate items list-pair by list-pair, then average."""
    total, n_ch = matrix.shape
    n_lists = total // n_words
    reshaped = matrix[:n_lists * n_words].reshape(n_lists, n_words, n_ch)
    if fixed_indices is not None:
        top_idx = fixed_indices
    else:
        top_idx = np.argsort(np.var(np.mean(reshaped, axis=0), axis=0))[::-1][:n_channels]
    data = reshaped[:, :, top_idx]
    np.random.seed(42)
    idx = np.random.permutation(n_lists)
    half = n_lists // 2
    A, B = data[idx[:half]], data[idx[half:half * 2]]
    pair = [np.corrcoef(la, lb)[:n_words, n_words:] for la in A for lb in B]
    return np.mean(pair, axis=0)


def second_order_rsa(iters_s0, iters_s1):
    """Relative representation: per-position row correlation of the two RDMs.

    In : per-iteration RDM stacks (n_iter, n, n). Out: (mean[n], sem[n]) over iters.
    """
    n_iter = min(iters_s0.shape[0], iters_s1.shape[0])
    n = iters_s0.shape[1]
    per = np.zeros((n_iter, n))
    for it in range(n_iter):
        for i in range(n):
            per[it, i] = pearsonr(iters_s0[it, i, :], iters_s1[it, i, :])[0]
    return per.mean(axis=0), per.std(axis=0, ddof=1) / np.sqrt(n_iter)


def second_order_matrix(iters_s0, iters_s1):
    """Full second-order matrix: every S0 RDM row vs every S1 RDM row, averaged over iters."""
    n_iter = min(iters_s0.shape[0], iters_s1.shape[0])
    n = iters_s0.shape[1]
    acc = np.zeros((n_iter, n, n))
    for it in range(n_iter):
        for i in range(n):
            for j in range(n):
                acc[it, i, j] = pearsonr(iters_s0[it, i, :], iters_s1[it, j, :])[0]
    return acc.mean(axis=0)


def words_gaps_second_order(rdm_w, rdm_g):
    """Second-order geometry: row i of the word RDM vs row j of the gap RDM."""
    n = rdm_w.shape[0]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            out[i, j] = pearsonr(rdm_w[i, :], rdm_g[j, :])[0]
    return out


@dataclass
class WordRSM:
    rsm_s0: np.ndarray
    rsm_s1: np.ndarray
    pearson_r: float
    cosine: float
    p_value: float


def word_rsm(words_s0, words_s1, n_channels=None, fixed_indices=None):
    """300x300 word RSM per session + cross-session consistency (Pearson + cosine).

    By default uses all channels. Pass `n_channels` to restrict to the top-N
    positional-variance channels (selected on words_s0) — this matches the book's
    "Global Neural Representation Stability" figure (100 channels) — or pass
    `fixed_indices` to reuse a specific channel set.
    """
    if fixed_indices is not None:
        idx = fixed_indices
    elif n_channels is not None:
        total, n_ch = words_s0.shape
        n_lists = total // 12
        reshaped = words_s0[:n_lists * 12].reshape(n_lists, 12, n_ch)
        pos_var = np.var(np.mean(reshaped, axis=0), axis=0)
        idx = np.argsort(pos_var)[::-1][:n_channels]
    else:
        idx = None
    if idx is not None:
        words_s0, words_s1 = words_s0[:, idx], words_s1[:, idx]

    rsm0 = np.corrcoef(words_s0)
    rsm1 = np.corrcoef(words_s1)
    triu = np.triu_indices(rsm0.shape[0], k=1)
    v0, v1 = rsm0[triu], rsm1[triu]
    r, p = pearsonr(v0, v1)
    return WordRSM(rsm0, rsm1, float(r), float(1 - _cosine_dist(v0, v1)), float(p))


@dataclass
class NeuralRSMs:
    words_s0: np.ndarray
    words_s1: np.ndarray
    gaps_s0: np.ndarray
    gaps_s1: np.ndarray


def neural_rsms(grand):
    """300x300 neural RSMs for words and gaps, per session (Notebook 2, Step 5)."""
    return NeuralRSMs(
        words_s0=np.corrcoef(grand.words_s0), words_s1=np.corrcoef(grand.words_s1),
        gaps_s0=np.corrcoef(grand.gaps_s0), gaps_s1=np.corrcoef(grand.gaps_s1),
    )


def second_order_layers(rsm_s0, rsm_s1, gpt2_rsms):
    """Second-order RSA: neural RSM vs each GPT-2 layer RSM (Pearson + Spearman).

    Compares the upper triangles (unique word pairs). Returns a dict with per-layer
    r and p, for both sessions and both metrics, plus the layer list.
    """
    layers = sorted(gpt2_rsms.keys())
    triu = np.triu_indices(rsm_s0.shape[0], k=1)
    v0, v1 = rsm_s0[triu], rsm_s1[triu]
    res = {"layers": layers}
    for m in ("pearson", "spearman"):
        for ses, vec in (("s0", v0), ("s1", v1)):
            res[f"{m}_{ses}"], res[f"{m}_p_{ses}"] = [], []
    for L in layers:
        g = gpt2_rsms[L][triu]
        for m, fn in (("pearson", pearsonr), ("spearman", spearmanr)):
            for ses, vec in (("s0", v0), ("s1", v1)):
                r, p = fn(vec, g)
                res[f"{m}_{ses}"].append(float(r))
                res[f"{m}_p_{ses}"].append(float(p))
    return res


def export_layer_table(result, out):
    """Write the per-layer RSA table (pearson/spearman r+p, both sessions) to CSV."""
    rows = []
    for i, L in enumerate(result["layers"]):
        rows.append({
            "layer": L,
            "pearson_r_s0": result["pearson_s0"][i], "pearson_p_s0": result["pearson_p_s0"][i],
            "pearson_r_s1": result["pearson_s1"][i], "pearson_p_s1": result["pearson_p_s1"][i],
            "spearman_r_s0": result["spearman_s0"][i], "spearman_p_s0": result["spearman_p_s0"][i],
            "spearman_r_s1": result["spearman_s1"][i], "spearman_p_s1": result["spearman_p_s1"][i],
        })
    import os
    os.makedirs(os.path.dirname(str(out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
