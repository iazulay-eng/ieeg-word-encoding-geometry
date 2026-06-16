# Data

This folder is where the dataset lives at runtime. It is **git-ignored** — nothing
here is committed except this README and `download_data.py`.

Point the code at the data with **one** of:

```bash
export IEEG_DATA_ROOT=/path/to/ieeg_data     # Linux / macOS
setx    IEEG_DATA_ROOT  D:\ieeg_data          # Windows
# or write the path into a one-line file:  <repo_root>/.ieeg_data_root
```

The Tier-B bundle (below) unpacks to this layout under `IEEG_DATA_ROOT`:

```
<IEEG_DATA_ROOT>/
├── dr-processed/<subject>/<session>/
│   ├── <sub>_<ses>_aligned_matrix_normratio.npy        # main neural input
│   ├── <sub>_<ses>_aligned_channels_raw.pkl            # channel names
│   ├── correlations_scores_normratio.csv               # per-channel reliability
│   └── <sub>_<ses>_task-FR1_acq-bipolar_ieeg.annot     # word events / labels
└── ds004789-download/<subject>/
    ├── <sub>_electrode_locations.csv                   # name,x,y,z,AAL_Label,Broad_ROI
    └── <session>/
        ├── beh/<sub>_<ses>_task-FR1_beh.tsv            # behavior (SPC)
        └── ieeg/<sub>_<ses>_..._electrodes.tsv         # MNI coordinates
```

> Notes: (1) filename casing is inconsistent across subjects (`normratio` vs `NormRatio`,
> `raw` vs `RAW`) — the loaders match case-insensitively. (2) The 48 `electrode_locations.csv`
> are also committed in this repo under `data/electrode_locations/`, so region info / brain
> maps work even without the bundle. (3) The brain map needs no atlas at runtime (AAL labels
> are pre-baked into the CSV).

## Three ways to get the data

| Tier | What you get | How | Re-run scope |
|---|---|---|---|
| **A** | electrode CSVs + behavioral CSVs + small derived matrices + figures | committed in the repo | reproduce figures, no download |
| **B** | the four files above for all 96 sessions (**~5 GB** tar) | `python data/download_data.py` | **full analysis, any config — no preprocessing** |
| **C** | nothing — regenerate from raw | run `preprocessing/` on raw `ds004789` (67 GB) | everything from scratch |

### Tier B — the processed bundle (recommended)

Hosted on **Google Drive** as a single ~5 GB tar (`ieeg-processed-min.tar`). Large Drive
downloads need `gdown`:

```bash
pip install gdown
python data/download_data.py            # uses the file id baked into the script
# or:  python data/download_data.py --id <drive_file_id> --dest /path/to/ieeg_data
```

It downloads the tar, extracts it to `dr-processed/` under your data root, and removes
the archive. After that, set `IEEG_DATA_ROOT` to that folder and run the notebooks —
**no raw data, no preprocessing.**
