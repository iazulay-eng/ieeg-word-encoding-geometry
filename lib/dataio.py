"""Loading processed per-session artifacts (case-insensitive) + electrode export.

The processed files use inconsistent casing across subjects (`normratio` vs
`NormRatio`, `raw` vs `RAW`), so all lookups here are case-insensitive.
"""

import glob
import os
import pickle

import numpy as np
import pandas as pd


def _find_ci(directory, filename):
    """Return the path of `filename` inside `directory`, matched case-insensitively."""
    target = filename.lower()
    if not os.path.isdir(directory):
        return None
    for path in glob.glob(os.path.join(directory, "*")):
        if os.path.basename(path).lower() == target:
            return path
    return None


def _region_map(subject, paths):
    """Map electrode (monopolar contact) name -> (AAL_Label, Broad_ROI).

    Looks first in the repo's committed data/electrode_locations/, then in the
    raw BIDS tree. Returns ({}, {}) if no file is found.
    """
    candidates = [
        paths.repo_root / "data" / "electrode_locations" / f"{subject}_electrode_locations.csv",
        paths.raw_dir / subject / f"{subject}_electrode_locations.csv",
    ]
    for csv in candidates:
        if csv.is_file():
            df = pd.read_csv(csv)
            name_col = "name" if "name" in df.columns else "label"
            aal = dict(zip(df[name_col], df.get("AAL_Label", pd.Series(dtype=str))))
            broad = dict(zip(df[name_col], df.get("Broad_ROI", pd.Series(dtype=str))))
            return aal, broad
    return {}, {}


def load_sessions(subjects, paths):
    """Load processed artifacts for every (subject, session) into memory.

    Per session reads (case-insensitively) from paths.processed_dir/<sub>/<ses>/:
      <sub>_<ses>_aligned_matrix_normratio.npy   (lists x channels x time)
      <sub>_<ses>_aligned_channels_raw.pkl       (channel-name list)
      correlations_scores_normratio.csv          (Channel, Avg_Correlation, SEM)

    Returns dict keyed "{sub}_{ses}" -> {
        matrix   : ndarray (lists, channels, time),
        channels : list[str],
        scores   : dict[channel -> Avg_Correlation],
        subject, session,
        aal, broad : dict[contact -> region]  (empty if no electrode CSV)
    }.
    """
    data = {}
    for sub, ses in subjects:
        base = os.path.join(str(paths.processed_dir), sub, ses)
        mat_p = _find_ci(base, f"{sub}_{ses}_aligned_matrix_normratio.npy")
        ch_p = _find_ci(base, f"{sub}_{ses}_aligned_channels_raw.pkl")
        sc_p = _find_ci(base, "correlations_scores_normratio.csv")
        if not (mat_p and ch_p and sc_p):
            continue

        with open(ch_p, "rb") as f:
            channels = pickle.load(f)
        scores_df = pd.read_csv(sc_p)
        scores = dict(zip(scores_df["Channel"], scores_df["Avg_Correlation"]))
        aal, broad = _region_map(sub, paths)

        data[f"{sub}_{ses}"] = {
            "matrix": np.load(mat_p),
            "channels": channels,
            "scores": scores,
            "subject": sub,
            "session": ses,
            "aal": aal,
            "broad": broad,
        }
    return data


def load_raw_matrix(subject, session, paths):
    """Load the RAW (non-normalized) aligned matrix + channels for PSTH dB plots.

    Returns (matrix (lists, channels, time), channels) or (None, None) if missing.
    """
    base = os.path.join(str(paths.processed_dir), subject, session)
    mat_p = _find_ci(base, f"{subject}_{session}_aligned_matrix_raw.npy")
    ch_p = _find_ci(base, f"{subject}_{session}_aligned_channels_raw.pkl")
    if not (mat_p and ch_p):
        return None, None
    with open(ch_p, "rb") as f:
        channels = pickle.load(f)
    return np.load(mat_p), channels


def export_selected_electrodes(df_features, scores, top_n, out):
    """Write a CSV of the top-N selected electrodes ranked by reliability score.

    df_features: DataFrame with columns subject, channel_name (the grand-matrix
    column map). scores: per-session {channel -> Avg_Correlation} dict-of-dicts or
    a flat {channel -> score}. Writes columns: subject, channel_name, score.
    """
    rows = []
    for _, r in df_features.iterrows():
        ch = r["channel_name"]
        sub = r["subject"]
        sc = None
        if isinstance(scores, dict) and sub + "_ses-0" in scores:
            sc = scores[sub + "_ses-0"].get(ch)
        elif isinstance(scores, dict):
            sc = scores.get(ch)
        rows.append({"subject": sub, "channel_name": ch, "score": sc})
    out_df = pd.DataFrame(rows)
    if "score" in out_df and out_df["score"].notna().any():
        out_df = out_df.sort_values("score", ascending=False)
    out_df = out_df.head(top_n)
    os.makedirs(os.path.dirname(str(out)), exist_ok=True)
    out_df.to_csv(out, index=False)
    return out_df
