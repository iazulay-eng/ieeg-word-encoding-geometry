"""Inter-trial reliability score (Step 3).

The score is the mean off-diagonal Pearson correlation of a channel's per-list
traces — how repeatable its response is across the 25 lists. By default it is
consumed from the `correlations_scores_normratio.csv` artifact (loaded into DATA);
`recompute=True` rebuilds it from the NormRatio matrices.
"""

import numpy as np


def inter_trial_scores(data, recompute=False):
    """Return per-session {channel -> reliability score}.

    In : data (from dataio.load_sessions); recompute -> rebuild from matrices.
    Out: dict keyed "{sub}_{ses}" -> {channel_name: score}.
    """
    out = {}
    for key, d in data.items():
        if not recompute:
            out[key] = d["scores"]
            continue

        matrix = d["matrix"]            # (lists, channels, time)
        channels = d["channels"]
        n_lists = matrix.shape[0]
        scores = {}
        for ci, ch in enumerate(channels):
            ch_data = matrix[:, ci, :]  # (lists, time)
            if n_lists < 2:
                scores[ch] = np.nan
                continue
            corr = np.corrcoef(ch_data)
            upper = corr[np.triu_indices_from(corr, k=1)]
            scores[ch] = float(np.nanmean(upper))
        out[key] = scores
    return out
