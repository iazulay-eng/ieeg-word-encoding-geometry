"""Behavioral serial-position curve, SPC (Step 1).

Ported from the recall-behavior notebook: parse the BIDS _beh.tsv files, map each
recalled word to its serial position (filtering intrusions and repetitions), and
build the per-subject recall-probability curve over the 12 positions.
"""

import os

import numpy as np
import pandas as pd

from . import dataio


def _beh_path(sub, ses, paths):
    base = os.path.join(str(paths.raw_dir), sub, ses, "beh")
    return dataio._find_ci(base, f"{sub}_{ses}_task-FR1_beh.tsv")


def compute_spc(subjects, paths, save=False):
    """Per-subject serial-position recall probabilities.

    Returns a tidy DataFrame (subject, session, Pos_1..Pos_12). If save=True,
    writes Subject_Level_SPC.csv and FR1_behavioral_summary.csv under results/.
    """
    spc_rows, summary_rows = [], []

    for sub, ses in subjects:
        path = _beh_path(sub, ses, paths)
        if path is None:
            continue
        df = pd.read_csv(path, sep="\t")
        df_task = df[df["list"] > 0]
        list_ids = df_task["list"].unique()
        recall = np.zeros((len(list_ids), 12))

        for i, lst in enumerate(list_ids):
            g = df_task[df_task["list"] == lst]
            presented = g[g["trial_type"] == "WORD"]
            word_to_pos = dict(zip(presented["item_name"], presented["serialpos"]))
            recalled = g[g["trial_type"] == "REC_WORD"]["item_name"].tolist()

            words, positions, seen = [], [], set()
            for w in recalled:
                if w in word_to_pos and w not in seen:   # drop intrusions + repetitions
                    pos = int(word_to_pos[w])
                    if 1 <= pos <= 12:
                        recall[i, pos - 1] = 1
                    words.append(str(w)); positions.append(str(pos)); seen.add(w)

            summary_rows.append({
                "Subject": sub, "Session": ses, "List_Number": int(lst),
                "Total_Recalled": len(words),
                "Recalled_Words": ";".join(words),
                "Serial_Positions": ";".join(positions),
            })

        vec = recall.mean(axis=0)
        spc_rows.append({"subject": sub, "session": ses,
                         **{f"Pos_{p+1}": vec[p] for p in range(12)}})

    spc = pd.DataFrame(spc_rows)

    if save:
        os.makedirs(str(paths.results_dir), exist_ok=True)
        spc.to_csv(paths.results_dir / "Subject_Level_SPC.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(summary_rows).to_csv(
            paths.results_dir / "FR1_behavioral_summary.csv", index=False, encoding="utf-8-sig")

    return spc
