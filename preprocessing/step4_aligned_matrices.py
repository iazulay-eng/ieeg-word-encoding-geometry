#!/usr/bin/env python
"""
generate_aligned_matrices.py
============================
Standalone version of the TEST NOTEBOOK's preprocessing cells 0.a / 0.b / 0.c
("Absolute Alignment with Fixed Gap Cropping" -> NormRatio -> inter-trial
correlation scores). Produces, for every subject/session:

    {sub}_{ses}_aligned_matrix_RAW.npy
    {sub}_{ses}_aligned_channels_RAW.pkl
    {sub}_{ses}_aligned_matrix_NormRatio.npy
    correlations_scores_NormRatio.csv

These are exactly the inputs that run_fast_analysis.py (the super-subject RSA
pipeline) reads. Data root is resolved from paths_and_constants (-> C:\\ieeg-data),
NOT the OneDrive folder. Per-channel PSTH plotting from the original cell is
intentionally omitted (not needed for the super-subject analysis).

Resumable: a session whose RAW matrix already exists is skipped.
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd
import mne

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths_and_constants import DATA_ROOT
from run_fast_analysis import SUBJECTS_TO_PROCESS

mne.set_log_level("ERROR")

PROC_DIR = os.path.join(DATA_ROOT, "dr-processed")
FIXED_GAP_SAMPLES = 50  # 0.5 s at 100 Hz  (matches "0.5 Gap" notebook)


# ---------------------------------------------------------------------------
# Cell 0.a : build RAW aligned matrix (absolute alignment, fixed gap cropping)
# ---------------------------------------------------------------------------
def process_subject_raw(subject, session, n_gap_samples_fixed=FIXED_GAP_SAMPLES):
    base_output_dir = os.path.join(PROC_DIR, subject, session)
    epo_path = os.path.join(base_output_dir, "gamma_c_60_160",
                            f"{subject}_{session}_task-FR1_acq-bipolar_ieeg_LIST_epo.fif")
    annot_path = os.path.join(base_output_dir,
                              f"{subject}_{session}_task-FR1_acq-bipolar_ieeg.annot")

    output_npy = os.path.join(base_output_dir, f"{subject}_{session}_aligned_matrix_RAW.npy")
    if os.path.exists(output_npy):
        print(f"  skip (RAW exists): {subject}_{session}")
        return True

    if not (os.path.exists(epo_path) and os.path.exists(annot_path)):
        print(f"  MISSING epo/annot: {subject}_{session}")
        return False

    try:
        epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
        epochs.drop_channels(epochs.info["bads"])
        data = epochs.get_data()  # (trials, channels, time) raw HFB power
        with open(annot_path, "rb") as f:
            annot_data = pickle.load(f)
        df_words = pd.DataFrame(annot_data["events"]["words"]).sort_values(["list", "onset"])
        unique_trials = df_words["list"].unique()
    except Exception as e:
        print(f"  ERROR loading {subject}_{session}: {e}")
        return False

    sfreq = epochs.info["sfreq"]
    n_word_samples = int(1.6 * sfreq)
    n_pre_buffer = int(1.5 * sfreq)
    n_post_word_12_gap_samples = int(0.15 * sfreq)
    channels_to_process = epochs.ch_names

    start_baseline_idx, end_baseline_idx = epochs.time_as_index([-1.5, 0])
    t0_idx = epochs.time_as_index(0)[0]

    canonical = {}
    for channel in channels_to_process:
        ch_idx = epochs.ch_names.index(channel)
        collected = []
        for show_idx in range(min(len(data), len(unique_trials))):
            signal = data[show_idx, ch_idx, :]
            trial_words_df = df_words[df_words["list"] == unique_trials[show_idx]]
            if len(trial_words_df) != 12:
                continue
            onsets = trial_words_df["onset"].values
            base_onset = onsets[0]
            stitched = [signal[start_baseline_idx:end_baseline_idx]]
            for i in range(12):
                rel = int((onsets[i] - base_onset) * sfreq)
                w_start = t0_idx + rel
                if w_start + n_word_samples > len(signal):
                    break
                stitched.append(signal[w_start:w_start + n_word_samples])
                gap_start = w_start + n_word_samples
                if i < 11:
                    next_rel = int((onsets[i + 1] - base_onset) * sfreq)
                    next_w_start = t0_idx + next_rel
                    gap = signal[gap_start:next_w_start][:n_gap_samples_fixed]
                    if len(gap) < n_gap_samples_fixed:
                        gap = np.pad(gap, (0, n_gap_samples_fixed - len(gap)), "edge")
                    stitched.append(gap)
                else:  # i == 11, final tail
                    end_idx = gap_start + n_post_word_12_gap_samples
                    final_gap = signal[gap_start:end_idx] if end_idx <= len(signal) else signal[gap_start:]
                    if len(final_gap) < n_post_word_12_gap_samples:
                        final_gap = np.pad(final_gap, (0, n_post_word_12_gap_samples - len(final_gap)), "edge")
                    stitched.append(final_gap)
            collected.append(np.concatenate(stitched))
        if collected:
            canonical[channel] = np.array(collected)

    final_channels = [ch for ch in channels_to_process if ch in canonical]
    if not final_channels:
        print(f"  no valid channels: {subject}_{session}")
        return False

    matrix_3d_raw = np.swapaxes(np.array([canonical[ch] for ch in final_channels]), 0, 1)
    np.save(output_npy, matrix_3d_raw)
    with open(os.path.join(base_output_dir, f"{subject}_{session}_aligned_channels_RAW.pkl"), "wb") as f:
        pickle.dump(final_channels, f)
    print(f"  RAW ok: {subject}_{session}  shape={matrix_3d_raw.shape}")
    return True


# ---------------------------------------------------------------------------
# Cell 0.b : single-trial ratio (fold-change) normalization
# ---------------------------------------------------------------------------
def normalize_ratio(subject, session):
    base_path = os.path.join(PROC_DIR, subject, session)
    raw_path = os.path.join(base_path, f"{subject}_{session}_aligned_matrix_RAW.npy")
    if not os.path.exists(raw_path):
        return False
    n_pre_buffer = 150  # 1.5 s at 100 Hz
    raw_mat = np.load(raw_path)
    baseline_means = np.maximum(np.mean(raw_mat[:, :, :n_pre_buffer], axis=2, keepdims=True), 1e-12)
    norm = raw_mat / baseline_means
    np.save(os.path.join(base_path, f"{subject}_{session}_aligned_matrix_NormRatio.npy"), norm)
    return True


# ---------------------------------------------------------------------------
# Cell 0.c : inter-trial correlation scores on NormRatio matrices
# ---------------------------------------------------------------------------
def correlation_scores(subject, session):
    base_path = os.path.join(PROC_DIR, subject, session)
    matrix_path = os.path.join(base_path, f"{subject}_{session}_aligned_matrix_NormRatio.npy")
    channels_path = os.path.join(base_path, f"{subject}_{session}_aligned_channels_RAW.pkl")
    if not (os.path.exists(matrix_path) and os.path.exists(channels_path)):
        return False
    norm_mat = np.load(matrix_path)
    with open(channels_path, "rb") as f:
        channel_names = pickle.load(f)
    n_trials = norm_mat.shape[0]
    if n_trials < 2:
        return False
    results = []
    for ch_idx, ch_name in enumerate(channel_names):
        corr_mat = np.corrcoef(norm_mat[:, ch_idx, :])
        upper = corr_mat[np.triu_indices_from(corr_mat, k=1)]
        results.append({"Channel": ch_name,
                        "Avg_Correlation": np.nanmean(upper),
                        "SEM": np.nanstd(upper) / np.sqrt(len(upper))})
    if not results:
        return False
    df = pd.DataFrame(results).sort_values("Avg_Correlation", ascending=False)
    df.to_csv(os.path.join(base_path, "correlations_scores_NormRatio.csv"), index=False)
    return True


def main():
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"Sessions to process: {len(SUBJECTS_TO_PROCESS)}")
    t0 = time.time()

    print("\n=== Stage A: RAW aligned matrices ===")
    ok_raw = sum(process_subject_raw(s, e) for s, e in SUBJECTS_TO_PROCESS)

    print("\n=== Stage B: NormRatio normalization ===")
    ok_norm = sum(normalize_ratio(s, e) for s, e in SUBJECTS_TO_PROCESS)

    print("\n=== Stage C: inter-trial correlation scores ===")
    ok_corr = sum(correlation_scores(s, e) for s, e in SUBJECTS_TO_PROCESS)

    print(f"\nDone in {time.time()-t0:.0f}s | RAW ok={ok_raw}  NormRatio ok={ok_norm}  Corr ok={ok_corr}")


if __name__ == "__main__":
    main()
