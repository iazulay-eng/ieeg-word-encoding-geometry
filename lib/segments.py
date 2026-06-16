"""Word/gap segment extraction + time-averaging (Step 5).

For each session, slice the per-channel word (1.6 s) and gap (0.5 s; last gap
0.15 s, NaN-padded) windows from the aligned matrix using the annotation word
order, then average within the requested sub-window to one scalar per channel.
"""

import os
import pickle

import numpy as np
import pandas as pd

from . import dataio

SFREQ = 100
N_WORD_SAMPLES = int(1.6 * SFREQ)      # 160
N_GAP_SAMPLES = int(0.5 * SFREQ)       # 50
N_LAST_GAP_SAMPLES = int(0.15 * SFREQ)  # 15
N_BASELINE_SAMPLES = int(1.5 * SFREQ)   # 150


def _load_words_df(annot_path):
    with open(annot_path, "rb") as f:
        annot = pickle.load(f)
    if "events" not in annot or "words" not in annot["events"]:
        return None, None, None
    df = pd.DataFrame(annot["events"]["words"])
    name_col = "word" if "word" in df.columns else ("item_name" if "item_name" in df.columns else "item")
    trial_col = "list" if "list" in df.columns else "trial"
    df = df.sort_values([trial_col, "onset"])
    return df, name_col, trial_col


def _extract(matrix, df_words, name_col, trial_col, kind):
    """Return (items, channels, time_window) segments + labels for words|gaps."""
    n_shows = matrix.shape[0]
    trials = df_words[trial_col].unique()
    segs, labels = [], []
    for i in range(min(len(trials), n_shows)):
        list_id = trials[i]
        if list_id <= 0:                       # skip practice lists
            continue
        list_df = df_words[df_words[trial_col] == list_id]
        if len(list_df) != 12:
            continue
        names = list_df[name_col].values
        for wi in range(12):
            base = N_BASELINE_SAMPLES + wi * (N_WORD_SAMPLES + N_GAP_SAMPLES)
            if kind == "words":
                start, end = base, base + N_WORD_SAMPLES
            else:
                start = base + N_WORD_SAMPLES
                end = start + (N_GAP_SAMPLES if wi < 11 else N_LAST_GAP_SAMPLES)
            if end > matrix.shape[2]:
                continue
            cut = matrix[i, :, start:end]
            if kind == "gaps" and wi == 11 and cut.shape[1] < N_GAP_SAMPLES:
                pad = N_GAP_SAMPLES - cut.shape[1]
                cut = np.concatenate((cut, np.full((cut.shape[0], pad), np.nan)), axis=1)
            segs.append(cut)
            labels.append(names[wi])
    if not segs:
        return None, None
    return np.stack(segs), np.array(labels)


def _time_average(segs, window):
    idx0, idx1 = int(window[0] * SFREQ), int(window[1] * SFREQ)
    idx1 = min(idx1, segs.shape[2])
    return np.nanmean(segs[:, :, idx0:idx1], axis=2)   # (items, channels)


def gap_segments_full(SEL, paths):
    """Per-session full (non-averaged) gap segments, for time-resolved/maintenance.

    Returns dict "{sub}_{ses}" -> {X (items, channels, gap_time), y}.
    """
    out = {}
    for key, d in SEL.items():
        parts = key.split("_")
        ses, sub = parts[-1], "_".join(parts[:-1])
        base = os.path.join(str(paths.processed_dir), sub, ses)
        annot = dataio._find_ci(base, f"{sub}_{ses}_task-FR1_acq-bipolar_ieeg.annot")
        if annot is None:
            continue
        df_words, name_col, trial_col = _load_words_df(annot)
        if df_words is None:
            continue
        g_segs, g_y = _extract(d["matrix"], df_words, name_col, trial_col, "gaps")
        if g_segs is None:
            continue
        out[key] = {"X": g_segs, "y": g_y}
    return out


def extract_words_and_gaps(SEL, paths, word_window=(0.0, 0.5), gap_window=(0.0, 0.5)):
    """Build per-session time-averaged word & gap matrices.

    Returns dict keyed "{sub}_{ses}" -> {words (items x ch), gaps (items x ch), y}.
    """
    out = {}
    for key, d in SEL.items():
        parts = key.split("_")
        ses = parts[-1]
        sub = "_".join(parts[:-1])
        base = os.path.join(str(paths.processed_dir), sub, ses)
        annot = dataio._find_ci(base, f"{sub}_{ses}_task-FR1_acq-bipolar_ieeg.annot")
        if annot is None:
            continue
        df_words, name_col, trial_col = _load_words_df(annot)
        if df_words is None:
            continue
        matrix = d["matrix"]
        w_segs, w_y = _extract(matrix, df_words, name_col, trial_col, "words")
        g_segs, g_y = _extract(matrix, df_words, name_col, trial_col, "gaps")
        if w_segs is None or g_segs is None:
            continue
        out[key] = {
            "words": _time_average(w_segs, word_window),
            "gaps": _time_average(g_segs, gap_window),
            "y": w_y,
        }
    return out
