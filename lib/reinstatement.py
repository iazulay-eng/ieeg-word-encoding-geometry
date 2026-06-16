"""Reinstatement: words-vs-gaps cross-RDM, time-resolved decay, maintenance (Steps 10-11).

Ported from the validated fast-analysis pipeline (cells 16c, 17e, 18, 21).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import pearsonr

from . import rsa, segments

SFREQ = 100


def cross_condition_rdm(grand_words, grand_gaps, fixed_indices, n_words=12):
    """Cross-condition RDM: word population profile i vs gap population profile j.

    The diagonal [i, i] is reinstatement — the word-i pattern showing up in gap i.
    Uses the mean-over-lists population matrices on the shared selected channels.
    Returns a (n_words, n_words) matrix.
    """
    pw = rsa.population_matrix(grand_words, fixed_indices, n_words)   # (12, N)
    pg = rsa.population_matrix(grand_gaps, fixed_indices, n_words)
    n = pw.shape[0]
    rdm = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if np.std(pw[i]) > 0 and np.std(pg[j]) > 0:
                rdm[i, j] = np.corrcoef(pw[i], pg[j])[0, 1]
    return rdm


def cross_session_consistency(cross_s0, cross_s1):
    """Second-order RSA of the cross-RDM across sessions: row i (S0) vs row j (S1).

    The diagonal asks whether the reinstatement *profile* of each position is
    reproduced across the two recording days. Returns a (n, n) matrix.
    """
    n = cross_s0.shape[0]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            out[i, j] = pearsonr(cross_s0[i, :], cross_s1[j, :])[0]
    return out


@dataclass
class TimeResolved:
    windows: dict                       # "t0_t1" -> {"s0": rdm, "s1": rdm}
    time_labels: list = field(default_factory=list)


def time_resolved_gap(SEL, paths, idx_s0, idx_s1, window_ms=50, n_windows=10):
    """Recompute the gap serial-position RDM in successive 50 ms windows.

    Slices the full gap segments into windows, builds the super-subject grand
    matrix per window, and computes the RDM on the locked word-channel indices.
    Returns a TimeResolved holding per-window S0/S1 RDMs (12x12, or 11x11 if the
    last position is all-NaN in that window).
    """
    gaps = segments.gap_segments_full(SEL, paths)
    subjects = sorted({k.split("_")[0] for k in gaps})
    windows, labels = {}, []

    for w in range(n_windows):
        t0, t1 = w * window_ms, (w + 1) * window_ms
        labels.append(f"{t0}-{t1}")
        i0, i1 = int(t0 / 1000 * SFREQ), int(t1 / 1000 * SFREQ)

        win = {}
        for key, d in gaps.items():
            X = d["X"]
            win[key] = np.nanmean(X[:, :, i0:min(i1, X.shape[2])], axis=2)

        l0, l1 = [], []
        for sub in subjects:
            k0, k1 = f"{sub}_ses-0", f"{sub}_ses-1"
            if k0 in win and k1 in win:
                l0.append(win[k0]); l1.append(win[k1])
        if not l0:
            continue

        rdm_s0 = rsa.rdm_with_fixed_features(np.hstack(l0), idx_s0)
        rdm_s1 = rsa.rdm_with_fixed_features(np.hstack(l1), idx_s1)
        if np.isnan(rdm_s0[11, 11]) or np.isnan(rdm_s1[11, 11]):
            rdm_s0, rdm_s1 = rdm_s0[:11, :11], rdm_s1[:11, :11]
        windows[f"{t0}_{t1}"] = {"s0": rdm_s0, "s1": rdm_s1}

    return TimeResolved(windows=windows, time_labels=labels)


@dataclass
class Maintenance:
    correlations_s0: dict               # position -> [r per window]
    correlations_s1: dict
    time_labels: list = field(default_factory=list)


def maintenance(grand_words_s0, grand_words_s1, SEL, paths, idx_s0, idx_s1,
                window_ms=50, n_windows=10, n_pos=12):
    """How far each position's word template persists into later gaps.

    For each 50 ms gap window, correlate every position's word population template
    with that position's mean gap profile in the window. Returns per-position
    correlation traces for both sessions.
    """
    pop_w0 = rsa.population_matrix(grand_words_s0, idx_s0, n_pos)   # (12, N)
    pop_w1 = rsa.population_matrix(grand_words_s1, idx_s1, n_pos)

    gaps = segments.gap_segments_full(SEL, paths)
    subjects = sorted({k.split("_")[0] for k in gaps})
    corr_s0 = {p: [] for p in range(n_pos)}
    corr_s1 = {p: [] for p in range(n_pos)}
    labels = []

    for w in range(n_windows):
        t0, t1 = w * window_ms, (w + 1) * window_ms
        labels.append(f"{t0}-{t1}")
        i0, i1 = int(t0 / 1000 * SFREQ), int(t1 / 1000 * SFREQ)

        for ses, pop_w, idx, corr in [("ses-0", pop_w0, idx_s0, corr_s0),
                                       ("ses-1", pop_w1, idx_s1, corr_s1)]:
            cols = []
            for sub in subjects:
                key = f"{sub}_{ses}"
                if key in gaps:
                    X = gaps[key]["X"]
                    cols.append(np.nanmean(X[:, :, i0:min(i1, X.shape[2])], axis=2))
            if not cols:
                continue
            grand = np.hstack(cols)[:, idx]
            n_lists = grand.shape[0] // n_pos
            mean_gaps = np.mean(grand[:n_lists * n_pos].reshape(n_lists, n_pos, -1), axis=0)
            for pos in range(n_pos):
                corr[pos].append(pearsonr(pop_w[pos], mean_gaps[pos])[0])

    return Maintenance(correlations_s0=corr_s0, correlations_s1=corr_s1, time_labels=labels)
