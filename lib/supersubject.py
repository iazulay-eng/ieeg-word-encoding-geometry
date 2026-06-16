"""Channel selection + super-subject grand matrices (Step 4).

Ported from the validated fast-analysis pipeline. Channels are kept by reliability
score (and optionally AAL region), intersected across both sessions, then z-scored
per channel over the whole session. Grand matrices concatenate subjects column-wise.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

_ALL = ("all", "All", "All Regions")

# AAL prefixes used for the broad regions that are not pre-tagged in Broad_ROI.
_PFC_PREFIXES = ("Frontal_Sup", "Frontal_Mid", "Frontal_Sup_Medial",
                 "Frontal_Med_Orb", "Frontal_Mid_Orb", "Frontal_Sup_Orb")


@dataclass
class Grand:
    words_s0: np.ndarray
    words_s1: np.ndarray
    gaps_s0: np.ndarray
    gaps_s1: np.ndarray
    features: pd.DataFrame
    word_labels: np.ndarray


def _top_channels(score_map, candidates, top_n, min_score):
    """Channels in `candidates` with score > min_score, ranked desc, top_n."""
    scored = [(c, score_map.get(c, np.nan)) for c in candidates]
    scored = [(c, s) for c, s in scored if pd.notna(s) and s > min_score]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [c for c, _ in scored[:top_n]]


def _roi_monopolars(aal_map, broad_map, region):
    """Monopolar contacts belonging to `region` (mirrors the pipeline's masks)."""
    contacts = []
    for name, aal in aal_map.items():
        aal = str(aal)
        if region == "Motor Cortex":
            hit = "Precentral" in aal
        elif region in ("Occipital", "Temporal"):
            hit = aal.startswith(region)
        elif region == "PFC":
            hit = aal.startswith(_PFC_PREFIXES)
        else:  # IFG, MTL, ... — use the pre-built Broad_ROI tag
            hit = broad_map.get(name) == region
        if hit:
            contacts.append(name)
    return set(contacts)


def select_channels(data, scores, region="all", top_n=180, min_score=-1, zscore=True):
    """Select channels per subject (intersected across sessions) and z-score.

    Returns (SEL, df_features):
      SEL          : dict "{sub}_{ses}" -> {matrix (lists, sel_ch, time), channels}
      df_features  : DataFrame(global_col_idx, subject, channel_name) in the
                     subject order used to build the grand matrices.
    """
    subjects = sorted({d["subject"] for d in data.values()})
    sel, feature_info, col = {}, [], 0

    for sub in subjects:
        k0, k1 = f"{sub}_ses-0", f"{sub}_ses-1"
        if k0 not in data or k1 not in data:
            continue
        d0, d1 = data[k0], data[k1]

        if region in _ALL:
            cand0, cand1 = d0["channels"], d1["channels"]
        else:
            roi = _roi_monopolars(d0["aal"], d0["broad"], region)
            cand0 = [c for c in d0["channels"] if any(p in roi for p in c.split("-"))]
            cand1 = [c for c in d1["channels"] if any(p in roi for p in c.split("-"))]

        top0 = _top_channels(scores[k0], cand0, top_n, min_score)
        top1 = _top_channels(scores[k1], cand1, top_n, min_score)
        common = sorted(set(top0).intersection(top1))
        if not common:
            continue

        ok = True
        staged = {}
        for key, d in [(k0, d0), (k1, d1)]:
            idx = [d["channels"].index(c) for c in common if c in d["channels"]]
            names = [c for c in common if c in d["channels"]]
            if not idx:
                ok = False
                break
            mat = d["matrix"][:, idx, :]
            if zscore:
                mean_c = np.mean(mat, axis=(0, 2), keepdims=True)
                std_c = np.std(mat, axis=(0, 2), keepdims=True)
                std_c[std_c == 0] = 1
                mat = (mat - mean_c) / std_c
            staged[key] = {"matrix": mat, "channels": names}
        if not ok:
            continue

        sel.update(staged)
        for i, name in enumerate(staged[k0]["channels"]):
            feature_info.append({"global_col_idx": col + i, "subject": sub, "channel_name": name})
        col += len(staged[k0]["channels"])

    return sel, pd.DataFrame(feature_info)


def build_grand_matrices(segs, df_features, align_vocab=False):
    """Concatenate per-subject time-averaged matrices into super-subject grand matrices.

    `segs` (from segments.extract_words_and_gaps) holds, per session key, the
    time-averaged word/gap matrices (items x channels) and word labels. Subjects
    are taken in df_features order; only those present with matching channel counts
    in both sessions are included.

    align_vocab=True keeps only sessions whose word order matches the reference
    (first session), guaranteeing every column shares the identical 300-word order
    — used by the GPT-2 notebook where the RSM is over word identities.
    """
    subjects = list(dict.fromkeys(df_features["subject"]))  # preserve order, unique
    lw0, lw1, lg0, lg1 = [], [], [], []
    word_labels = None

    for sub in subjects:
        k0, k1 = f"{sub}_ses-0", f"{sub}_ses-1"
        if k0 not in segs or k1 not in segs:
            continue
        w0, w1 = segs[k0]["words"], segs[k1]["words"]
        g0, g1 = segs[k0]["gaps"], segs[k1]["gaps"]
        if w0.shape[1] != w1.shape[1]:
            continue
        if align_vocab:
            if word_labels is None:
                word_labels = segs[k0]["y"]
            if not (np.array_equal(segs[k0]["y"], word_labels)
                    and np.array_equal(segs[k1]["y"], word_labels)):
                continue
        lw0.append(w0); lw1.append(w1); lg0.append(g0); lg1.append(g1)
        if word_labels is None:
            word_labels = segs[k0]["y"]

    return Grand(
        words_s0=np.hstack(lw0), words_s1=np.hstack(lw1),
        gaps_s0=np.hstack(lg0), gaps_s1=np.hstack(lg1),
        features=df_features, word_labels=word_labels,
    )
