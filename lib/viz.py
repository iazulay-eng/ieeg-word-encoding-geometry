"""All plotting helpers. The notebooks call only these; each saves a figure.

Plotting only — the numbers come from the analysis modules. Uses a non-interactive
backend so the notebooks run headless.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from . import dataio

SFREQ = 100
N_BASELINE = 150
SPC_COLS = [f"Pos_{i}" for i in range(1, 13)]


def _save(out):
    out = str(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()


def _diag(m):
    return np.array([m[i, i] for i in range(m.shape[0])])


# ====================================================================
# Step 1 — Behavioral SPC
# ====================================================================

def _spc_means(spc):
    s0 = spc[spc["session"] == "ses-0"][SPC_COLS]
    s1 = spc[spc["session"] == "ses-1"][SPC_COLS]
    return s0, s1


def plot_spc_curve(spc, out):
    s0, s1 = _spc_means(spc)
    x = np.arange(1, 13)
    plt.figure(figsize=(10, 6))
    for df, lab, c, m in [(s0, "Session 0", "tab:blue", "o"), (s1, "Session 1", "darkorange", "s")]:
        if len(df):
            plt.plot(x, df.mean().values, marker=m, lw=2, label=lab, color=c)
    plt.title("Serial Position Curve (group average)", fontsize=15)
    plt.xlabel("Serial position (1-12)"); plt.ylabel("Probability of recall")
    plt.xticks(x); plt.ylim(0, 1.05); plt.grid(True, ls="--", alpha=0.6); plt.legend()
    _save(out)


def plot_spc_bars(spc, out):
    from scipy.stats import sem
    s0, s1 = _spc_means(spc)
    x = np.arange(1, 13); w = 0.38
    plt.figure(figsize=(12, 6))
    if len(s0):
        plt.bar(x - w/2, s0.mean().values, w, yerr=s0.apply(sem).values, capsize=4,
                color="royalblue", edgecolor="black", alpha=0.85, label="Session 0")
    if len(s1):
        plt.bar(x + w/2, s1.mean().values, w, yerr=s1.apply(sem).values, capsize=4,
                color="darkorange", edgecolor="black", alpha=0.85, label="Session 1")
    plt.title("Serial Position Curve (bar, +/- SEM)", fontsize=15)
    plt.xlabel("Serial position (1-12)"); plt.ylabel("Probability of recall")
    plt.xticks(x); plt.ylim(0, 1.1); plt.grid(axis="y", ls="--", alpha=0.6); plt.legend()
    _save(out)


# ====================================================================
# Step 3 / 6 — PSTH
# ====================================================================

def plot_psth(data, scores, top_k=5, out=None):
    """Mean (over trials) NormRatio trace of the top_k channels in the first session."""
    key = sorted(data.keys())[0]
    d = data[key]
    sc = scores[key]
    ranked = sorted(d["channels"], key=lambda c: sc.get(c, -np.inf), reverse=True)[:top_k]
    t = (np.arange(d["matrix"].shape[2]) - N_BASELINE) / SFREQ
    plt.figure(figsize=(16, 5))
    for ch in ranked:
        ci = d["channels"].index(ch)
        plt.plot(t, np.nanmean(d["matrix"][:, ci, :], axis=0), lw=1.2, label=f"{ch} (r={sc.get(ch, float('nan')):.2f})")
    plt.axvline(0, color="k", lw=1); plt.title(f"PSTH — top {top_k} channels ({key})")
    plt.xlabel("Time (s)"); plt.ylabel("NormRatio (mean over lists)")
    plt.legend(fontsize=8, ncol=2); _save(out)


def plot_population_psth(SEL, df_features, scores, top_n=100, session="ses-0", mode="raw_then_db", out=None):
    """Grand-average PSTH of the top_n channels (by score), average-first-then-dB.

    Loads the RAW aligned matrices for the selected channels, averages over lists,
    converts each channel to dB vs its baseline, then averages across channels.
    """
    # rank the selected channels by score within the session
    ranked = []
    for _, r in df_features.iterrows():
        key = f"{r['subject']}_{session}"
        ranked.append((r["subject"], r["channel_name"], scores.get(key, {}).get(r["channel_name"], -np.inf)))
    ranked = sorted(ranked, key=lambda t: t[2], reverse=True)[:top_n]

    by_sub = {}
    for sub, ch, _ in ranked:
        by_sub.setdefault(sub, []).append(ch)

    traces = []
    paths = _PATHS_HINT[0]
    for sub, chs in by_sub.items():
        raw, raw_chs = dataio.load_raw_matrix(sub, session, paths)
        if raw is None:
            continue
        mean_over_trials = np.mean(raw, axis=0)
        for ch in chs:
            if ch not in raw_chs:
                continue
            tr = mean_over_trials[raw_chs.index(ch), :]
            base = np.mean(tr[:N_BASELINE]); base = base if base > 0 else 1e-12
            traces.append(10 * np.log10(np.maximum(tr, 1e-12) / base))
    if not traces:
        return
    traces = np.array(traces)
    t = (np.arange(traces.shape[1]) - N_BASELINE) / SFREQ
    mean_db = traces.mean(axis=0)
    sem_db = traces.std(axis=0) / np.sqrt(len(traces))
    plt.figure(figsize=(16, 5))
    plt.fill_between(t, mean_db - sem_db, mean_db + sem_db, color="#2ca02c", alpha=0.3)
    plt.plot(t, mean_db, color="#2ca02c", lw=2)
    plt.axhline(0, color="k", lw=1.2)
    plt.title(f"Grand-average PSTH — top {top_n} channels ({session})")
    plt.xlabel("Time (s)"); plt.ylabel("Activity (dB)")
    _save(out)


# population PSTH needs paths to load raw matrices; the notebook sets this once.
_PATHS_HINT = [None]


def set_paths(paths):
    """Give viz access to PATHS (for plot_population_psth's raw-matrix loading)."""
    _PATHS_HINT[0] = paths


# ====================================================================
# Step 7 — Word RSM
# ====================================================================

def plot_word_rsm(rsm, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, m, ttl in zip(axes, [rsm.rsm_s0, rsm.rsm_s1], ["Session 0", "Session 1"]):
        im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"Neural word RSM — {ttl}")
        ax.set_xlabel("Words"); ax.set_ylabel("Words")
        fig.colorbar(im, ax=ax, label="Pearson r")
    plt.suptitle(f"Word-identity RSM (cross-session r={rsm.pearson_r:.3f}, cos={rsm.cosine:.3f})", fontsize=14)
    plt.tight_layout(); _save(out)


def plot_rsm_consistency_scatter(rsm, out):
    triu = np.triu_indices(rsm.rsm_s0.shape[0], k=1)
    v0, v1 = rsm.rsm_s0[triu], rsm.rsm_s1[triu]
    plt.figure(figsize=(7, 7))
    if len(v0) > 20000:
        np.random.seed(0)
        idx = np.random.choice(len(v0), 20000, replace=False)
        v0p, v1p = v0[idx], v1[idx]
    else:
        v0p, v1p = v0, v1
    plt.scatter(v0p, v1p, alpha=0.1, s=2, color="darkblue")
    plt.plot([-1, 1], [-1, 1], "k--", alpha=0.5, label="identity")
    m, b = np.polyfit(v0, v1, 1)
    plt.plot([-1, 1], [m * -1 + b, m * 1 + b], "r", lw=2, label=f"fit (slope={m:.2f})")
    plt.xlim(-1, 1); plt.ylim(-1, 1); plt.grid(True, alpha=0.3); plt.legend()
    plt.title(f"Cross-session consistency (r={rsm.pearson_r:.3f}, cos={rsm.cosine:.3f})")
    plt.xlabel("Similarity in Session 0"); plt.ylabel("Similarity in Session 1")
    _save(out)


def plot_word_rsm_zoom(rsm, n=50, labels=None, out=None):
    n = min(n, rsm.rsm_s0.shape[0])
    ticks = [str(x) for x in (labels[:n] if labels is not None else range(1, n + 1))]
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    for ax, m, ttl in zip(axes, [rsm.rsm_s0, rsm.rsm_s1], ["Session 0", "Session 1"]):
        im = ax.imshow(m[:n, :n], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"{ttl}: first {n} words")
        ax.set_xticks(range(n)); ax.set_xticklabels(ticks, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(ticks, fontsize=7)
        fig.colorbar(im, ax=ax)
    plt.tight_layout(); _save(out)


# ====================================================================
# Step 8 — serial-position RDM + decoding diagonal
# ====================================================================

def plot_rdm(rdm_s0, rdm_s1, title="RDM", out=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, m, ttl in zip(axes, [rdm_s0, rdm_s1], ["Session 0", "Session 1"]):
        sns.heatmap(m, annot=True, fmt=".2f", annot_kws={"size": 7}, cmap="RdBu_r",
                    vmin=-1, vmax=1, ax=ax,
                    xticklabels=range(1, m.shape[0] + 1), yticklabels=range(1, m.shape[0] + 1))
        ax.set_title(f"{title} — {ttl}")
    plt.tight_layout(); _save(out)


def plot_decoding_diagonal(rdm_s0, rdm_s1, out=None):
    d0, d1 = _diag(rdm_s0), _diag(rdm_s1)
    x = np.arange(1, len(d0) + 1); w = 0.38
    plt.figure(figsize=(13, 5))
    plt.bar(x - w/2, d0, w, label="Session 0", color="#1f77b4", edgecolor="black")
    plt.bar(x + w/2, d1, w, label="Session 1", color="#ff7f0e", edgecolor="black")
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.title("Decoding diagonal (within-position reliability)")
    plt.xlabel("Serial position"); plt.ylabel("Pearson r"); plt.xticks(x); plt.legend()
    _save(out)


def plot_decoding_diagonal_errorbars(iters_s0, iters_s1, out=None):
    n = iters_s0.shape[1]
    d0 = np.array([[iters_s0[it, i, i] for i in range(n)] for it in range(iters_s0.shape[0])])
    d1 = np.array([[iters_s1[it, i, i] for i in range(n)] for it in range(iters_s1.shape[0])])
    x = np.arange(1, n + 1); w = 0.38
    plt.figure(figsize=(13, 5))
    plt.bar(x - w/2, d0.mean(0), w, yerr=d0.std(0, ddof=1)/np.sqrt(len(d0)), capsize=4,
            label="Session 0", color="#1f77b4", edgecolor="black")
    plt.bar(x + w/2, d1.mean(0), w, yerr=d1.std(0, ddof=1)/np.sqrt(len(d1)), capsize=4,
            label="Session 1", color="#ff7f0e", edgecolor="black")
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.title("Decoding diagonal (+/- SEM over split-half iterations)")
    plt.xlabel("Serial position"); plt.ylabel("Pearson r"); plt.xticks(x); plt.legend()
    _save(out)


# ====================================================================
# Step 9 — second-order RSA
# ====================================================================

def plot_relative_representation(words, words_sem, gaps, gaps_sem, out=None):
    xw = np.arange(1, len(words) + 1); xg = np.arange(1, len(gaps) + 1); w = 0.4
    plt.figure(figsize=(16, 7))
    plt.bar(xw - w/2, words, w, yerr=words_sem, capsize=4, color="#1f77b4", edgecolor="black", label="Words (S0 vs S1)")
    plt.bar(xg + w/2, gaps, w, yerr=gaps_sem, capsize=4, color="#ff7f0e", edgecolor="black", label="Gaps (S0 vs S1)")
    plt.axhline(0, color="k", ls="--", lw=1.2)
    plt.title("Relative representation (second-order RSA, cross-session)")
    plt.xlabel("Serial position"); plt.ylabel("Row correlation (Pearson r)")
    plt.xticks(xw); plt.legend(); _save(out)


def plot_second_order_matrices(so_words, so_gaps, out=None):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, m, ttl in zip(axes, [so_words, so_gaps], ["Words", "Gaps"]):
        sns.heatmap(m, annot=True, fmt=".2f", annot_kws={"size": 8}, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"Second-order RSA — {ttl}"); ax.set_xlabel("Session 1"); ax.set_ylabel("Session 0")
    plt.tight_layout(); _save(out)


def plot_second_order_diagonals(so_words, so_gaps, out=None):
    dw, dg = _diag(so_words), _diag(so_gaps)
    xw = np.arange(1, len(dw) + 1); xg = np.arange(1, len(dg) + 1); w = 0.4
    plt.figure(figsize=(16, 7))
    plt.bar(xw - w/2, dw, w, color="#1f77b4", edgecolor="black", label="Words diagonal")
    plt.bar(xg + w/2, dg, w, color="#ff7f0e", edgecolor="black", label="Gaps diagonal")
    plt.axhline(0, color="k", ls="--", lw=1.2)
    plt.title("Second-order RSA — diagonals"); plt.xlabel("Serial position"); plt.ylabel("Pearson r")
    plt.xticks(xw); plt.legend(); _save(out)


# ====================================================================
# Step 10 — reinstatement (words vs gaps)
# ====================================================================

def _cross_panels(cross_list):
    mats = [("Session 0", cross_list[0]), ("Session 1", cross_list[1])]
    mean = np.nanmean([cross_list[0], cross_list[1]], axis=0)
    mats.append(("Average", mean))
    return mats, mean


def plot_cross_rdm(cross_list, out=None):
    mats, mean = _cross_panels(cross_list)
    vmax = np.nanpercentile(np.abs(mean), 98)
    n = mean.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, (ttl, m) in zip(axes, mats):
        sns.heatmap(m, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-vmax, vmax=vmax, ax=ax,
                    xticklabels=range(1, n + 1), yticklabels=range(1, n + 1))
        ax.set_title(ttl); ax.set_xlabel("Gap position"); ax.set_ylabel("Word position")
    plt.suptitle("Cross-RDM: Words x Gaps (diagonal = reinstatement)", fontsize=14, y=1.02)
    plt.tight_layout(); _save(out)


def plot_cross_rdm_barplot(cross_list, out=None):
    _, mean = _cross_panels(cross_list)
    n = mean.shape[0]
    diag = _diag(mean)
    off = np.array([(mean[i].sum() - mean[i, i]) / (n - 1) for i in range(n)])
    x = np.arange(1, n + 1); w = 0.35
    plt.figure(figsize=(13, 5))
    plt.bar(x - w/2, diag, w, label="Diagonal (reinstatement)", color="#d62728")
    plt.bar(x + w/2, off, w, label="Off-diagonal mean", color="#aec7e8")
    plt.axhline(0, color="k", ls="--", lw=1.2)
    plt.title("Cross-RDM: diagonal vs off-diagonal (avg across sessions)")
    plt.xlabel("Serial position"); plt.ylabel("Pearson r"); plt.xticks(x); plt.legend()
    _save(out)


def plot_cross_session_consistency(consistency, out=None):
    n = consistency.shape[0]
    vmax = np.nanpercentile(np.abs(consistency), 98)
    plt.figure(figsize=(9, 8))
    sns.heatmap(consistency, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                xticklabels=range(1, n + 1), yticklabels=range(1, n + 1))
    plt.title("Second-order RSA: cross-RDM S0 x S1\n(diagonal = reinstatement consistent across sessions)")
    plt.xlabel("Position — Session 1"); plt.ylabel("Position — Session 0")
    _save(out)


def plot_words_gaps_second_order(so_list, out=None):
    mean = np.nanmean(so_list, axis=0)
    panels = [("Session 0", so_list[0]), ("Session 1", so_list[1]), ("Average", mean)]
    vmax = np.nanpercentile(np.abs(mean), 98); n = mean.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, (ttl, m) in zip(axes, panels):
        sns.heatmap(m, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-vmax, vmax=vmax, ax=ax,
                    xticklabels=range(1, n + 1), yticklabels=range(1, n + 1))
        ax.set_title(ttl); ax.set_xlabel("Gap position"); ax.set_ylabel("Word position")
    plt.suptitle("Second-order RSA: Words x Gaps geometry", fontsize=13, y=1.02)
    plt.tight_layout(); _save(out)


def plot_words_gaps_second_order_diagonal(so_list, out=None):
    d0, d1 = _diag(so_list[0]), _diag(so_list[1])
    x = np.arange(1, len(d0) + 1); w = 0.38
    plt.figure(figsize=(14, 6))
    plt.bar(x - w/2, d0, w, label="Session 0", color="#1f77b4", edgecolor="black")
    plt.bar(x + w/2, d1, w, label="Session 1", color="#ff7f0e", edgecolor="black")
    plt.axhline(0, color="k", ls="--", lw=1.2)
    plt.title("Second-order RSA: Words x Gaps — diagonal per position")
    plt.xlabel("Serial position"); plt.ylabel("Pearson r (diagonal)"); plt.xticks(x); plt.legend()
    _save(out)


# ====================================================================
# Step 11 — time-resolved + maintenance
# ====================================================================

def _row_corr(rdm_s0, rdm_s1):
    n = rdm_s0.shape[0]
    return np.array([pearsonr(rdm_s0[i], rdm_s1[i])[0] for i in range(n)])


def plot_time_resolved_rdms(tr, out_dir=None):
    for key, d in tr.windows.items():
        t0, t1 = key.split("_")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, m, ses in zip(axes, [d["s0"], d["s1"]], ["S0", "S1"]):
            sns.heatmap(m, annot=True, fmt=".2f", annot_kws={"size": 7}, cmap="RdBu_r",
                        vmin=-1, vmax=1, ax=ax,
                        xticklabels=range(1, m.shape[0] + 1), yticklabels=range(1, m.shape[0] + 1))
            ax.set_title(f"Gap RDM {ses} ({t0}-{t1} ms)")
        plt.tight_layout(); _save(os.path.join(str(out_dir), f"gap_rdm_{t0}_{t1}ms.png"))


def plot_time_resolved_relrep_bars(tr, out_dir=None):
    for key, d in tr.windows.items():
        t0, t1 = key.split("_")
        rr = _row_corr(d["s0"], d["s1"])
        x = np.arange(1, len(rr) + 1)
        plt.figure(figsize=(14, 6))
        plt.bar(x, rr, color="#ff7f0e", edgecolor="black")
        plt.axhline(0, color="k", ls="--", lw=1.2)
        plt.title(f"Relative representation in gap window {t0}-{t1} ms")
        plt.xlabel("Serial position"); plt.ylabel("Row correlation (Pearson r)"); plt.xticks(x)
        _save(os.path.join(str(out_dir), f"relrep_{t0}_{t1}ms.png"))


def _time_resolved_matrix(tr):
    keys = sorted(tr.windows.keys(), key=lambda k: int(k.split("_")[0]))
    cols, labels = [], []
    for k in keys:
        rr = _row_corr(tr.windows[k]["s0"], tr.windows[k]["s1"])
        if len(rr) < 12:
            rr = np.append(rr, [np.nan] * (12 - len(rr)))
        cols.append(rr); labels.append(k.replace("_", "-"))
    return np.array(cols).T, labels   # (12 positions, n_windows)


def plot_time_resolved_heatmap(tr, out=None):
    mat, labels = _time_resolved_matrix(tr)
    df = pd.DataFrame(mat, columns=labels, index=range(1, 13))
    plt.figure(figsize=(16, 8))
    ax = sns.heatmap(df, mask=df.isnull(), annot=True, fmt=".2f", cmap="RdBu_r",
                     vmin=np.nanmin(mat), vmax=np.nanmax(mat),
                     linewidths=0.5, linecolor="gray", cbar_kws={"label": "Relative representation (Pearson r)"})
    ax.set_facecolor("whitesmoke")
    plt.title("Time-resolved relative representation during gap (S0 vs S1)", fontsize=15)
    plt.xlabel("Time window in gap (ms)"); plt.ylabel("Serial position")
    plt.yticks(rotation=0); plt.xticks(rotation=45, ha="right"); _save(out)


def plot_decay_trendlines(tr, out=None):
    mat, labels = _time_resolved_matrix(tr)
    x = np.array([(int(a) + int(b)) / 2 for a, b in (l.split("-") for l in labels)])
    df = pd.DataFrame(mat, index=range(1, 13))
    groups = [("Primacy (1-3)", df.loc[1:3].mean().values, "#d62728", "o"),
              ("Middle (4-9)", df.loc[4:9].mean().values, "#7f7f7f", "s"),
              ("Recency (10-11)", df.loc[10:11].mean().values, "#1f77b4", "^")]
    plt.figure(figsize=(12, 6))
    for lab, y, c, m in groups:
        plt.plot(x, y, marker=m, ls="", color=c, alpha=0.4, ms=7)
        sl, ic = np.polyfit(x, y, 1)
        plt.plot(x, sl * x + ic, color=c, lw=3.5, label=f"{lab} (trend)")
    plt.axhline(0, color="k", ls="--", lw=1.5, alpha=0.7)
    plt.title("Linear trend of memory-trace decay during the gap", fontsize=15)
    plt.xlabel("Time after word disappearance (ms)"); plt.ylabel("Correlation (Pearson r)")
    plt.xticks(x, labels, rotation=45, ha="right"); plt.legend(); plt.grid(True, ls=":", alpha=0.6)
    _save(out)


def plot_maintenance(maint, out_dir=None):
    for pos in maint.correlations_s0:
        rp = pos + 1
        df = pd.DataFrame({"Time Window (ms)": maint.time_labels, "Session 0": maint.correlations_s0[pos]})
        has_s1 = len(maint.correlations_s1.get(pos, [])) > 0
        if has_s1:
            df["Session 1"] = maint.correlations_s1[pos]
        dm = df.melt(id_vars="Time Window (ms)", var_name="Session", value_name="Correlation")
        plt.figure(figsize=(14, 7))
        palette = ["#1f77b4", "#ff7f0e"] if has_s1 else ["#1f77b4"]
        ax = sns.barplot(data=dm, x="Time Window (ms)", y="Correlation", hue="Session",
                         palette=palette, edgecolor="black", alpha=0.9)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", padding=3, fontsize=9)
        plt.title(f"Maintenance of Word {rp} trace during Gap {rp}", fontsize=15)
        plt.xlabel(f"Time window within Gap {rp} (ms)"); plt.ylabel("Pearson r")
        plt.axhline(0, color="k", ls="--", lw=1.2); plt.xticks(rotation=45, ha="right")
        _save(os.path.join(str(out_dir), f"word{rp}_gap{rp}_maintenance.png"))


# ====================================================================
# Step 12 — decoding confusion
# ====================================================================

def plot_decoding_confusion(acc_rdm, acc_raw, chance=1/12, out=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (acc, ttl, cmap) in zip(axes, [(acc_rdm, "RDM space", "Blues"),
                                           (acc_raw, "Raw-activity space", "Reds")]):
        n = acc.confusion.shape[0]
        sns.heatmap(acc.confusion, ax=ax, cmap=cmap, cbar=False, linewidths=1, linecolor="black", square=True,
                    xticklabels=range(1, n + 1), yticklabels=range(1, n + 1))
        ax.set_title(f"{ttl}: {acc.accuracy*100:.0f}% (chance {chance*100:.1f}%)")
        ax.set_xlabel("Predicted position"); ax.set_ylabel("True position")
    plt.tight_layout(); _save(out)


# ====================================================================
# Notebook 2 — Neural <-> GPT-2
# ====================================================================

def plot_neural_rsm(rsm_s0, rsm_s1, title="Neural RSM", out=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, m, ses in zip(axes, [rsm_s0, rsm_s1], ["Session 0", "Session 1"]):
        im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"{title} — {ses}"); fig.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout(); _save(out)


def plot_gpt2_rsms(gpt2_rsms, layers=(0, 6, 12), out=None):
    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5))
    for ax, L in zip(np.atleast_1d(axes), layers):
        im = ax.imshow(gpt2_rsms[L], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"GPT-2 RSM — layer {L}"); fig.colorbar(im, ax=ax)
    plt.tight_layout(); _save(out)


def plot_rsa_layer_profile(result, title="Neural-GPT-2 RSA", out=None):
    layers = result["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, metric in zip(axes, ["pearson", "spearman"]):
        ax.plot(layers, result[f"{metric}_s0"], "b-o", lw=2, label="Session 0")
        ax.plot(layers, result[f"{metric}_s1"], "r-s", lw=2, label="Session 1")
        ax.axhline(0, color="gray", ls="--"); ax.set_title(f"{title} — {metric.title()}")
        ax.set_xlabel("GPT-2 layer"); ax.set_ylabel(f"{metric.title()} r"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); _save(out)


def plot_permutation_band(perm, out=None):
    layers = perm["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, ses in zip(axes, ["s0", "s1"]):
        obs = perm[f"observed_{ses}"]; null = perm[f"null_{ses}"]
        nm, nsd = null.mean(0), null.std(0)
        ax.fill_between(layers, nm - 2*nsd, nm + 2*nsd, color="gray", alpha=0.25, label="null +/- 2 SD")
        ax.plot(layers, nm, color="gray", ls="--", label="null mean")
        ax.plot(layers, obs, "b-o", lw=2, label="observed")
        for i, p in enumerate(perm[f"p_{ses}"]):
            if p < 0.05:
                ax.scatter(layers[i], obs[i], s=130, color="red", marker="*", zorder=5)
        ax.axhline(0, color="k", ls=":"); ax.set_title(f"Permutation test — {ses.upper()}")
        ax.set_xlabel("GPT-2 layer"); ax.set_ylabel("Spearman r"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); _save(out)


def plot_permutation_histogram(perm, out=None):
    obs0 = perm["observed_s0"]; null0 = perm["null_s0"]
    best = int(np.argmax((np.array(perm["observed_s0"]) + np.array(perm["observed_s1"])) / 2))
    plt.figure(figsize=(8, 5))
    plt.hist(null0[:, best], bins=50, color="steelblue", alpha=0.7, edgecolor="white", label="null")
    plt.axvline(obs0[best], color="red", lw=2.5, label=f"observed (layer {best})")
    plt.axvline(0, color="k", ls=":")
    plt.title(f"Permutation null — best layer {best}"); plt.xlabel("Spearman r"); plt.ylabel("count"); plt.legend()
    _save(out)
