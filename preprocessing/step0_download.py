#!/usr/bin/env python
"""
download_openneuro_subjects.py
==============================================================================
Download the 48 subjects that make up the "super subject" from OpenNeuro,
dataset ds004789 (https://openneuro.org/datasets/ds004789/versions/3.1.0).

This is the RAW BIDS data (ds004789-download/). Everything under dr-processed/
is then generated locally by your own preprocessing pipeline
(generate_annotations.py -> process_raw_signals.py -> generate_epoch_files.py),
and the per-subject NormRatio matrices / RSA "super subject" are built on top
of that.

The subject list is taken verbatim from `subjects_to_process` in
"Test notebook (0.5 Gap).ipynb" (the two commented-out subjects R1113T and
R1341T are intentionally excluded).

Setup (once):
    pip install openneuro-py

Run (from the project root, the folder that contains mycode/):
    python mycode/download_openneuro_subjects.py
"""

import os
import sys
import openneuro

# --- the 48 subjects (from the Test notebook's subjects_to_process) ----------
SUBJECTS = [
    "sub-R1001P", "sub-R1002P", "sub-R1010J", "sub-R1051J", "sub-R1054J",
    "sub-R1060M", "sub-R1065J", "sub-R1075J", "sub-R1076D", "sub-R1092J",
    "sub-R1098D", "sub-R1108J", "sub-R1118N", "sub-R1123C", "sub-R1136N",
    "sub-R1145J", "sub-R1151E", "sub-R1153T", "sub-R1154D", "sub-R1156D",
    "sub-R1161E", "sub-R1166D", "sub-R1168T", "sub-R1173J", "sub-R1189M",
    "sub-R1195E", "sub-R1196N", "sub-R1200T", "sub-R1201P", "sub-R1223E",
    "sub-R1234D", "sub-R1243T", "sub-R1283T", "sub-R1297T", "sub-R1292E",
    "sub-R1299T", "sub-R1310J", "sub-R1315T", "sub-R1317D", "sub-R1328E",
    "sub-R1332M", "sub-R1334T", "sub-R1338T", "sub-R1346T", "sub-R1350D",
    "sub-R1354E", "sub-R1355T", "sub-R1425D",
]

DATASET    = "ds004789"
TAG        = "3.1.0"   # pin to the version the super subject was built on
# Land the data where paths_and_constants.py expects it (BASE_FOLDER). This
# honours the IEEG_DATA_ROOT override / .ieeg_data_root file, so the data can
# live OUTSIDE the OneDrive-synced project folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths_and_constants import BASE_FOLDER
TARGET_DIR = BASE_FOLDER

# The super-subject analysis only ever uses ses-0 and ses-1, so we skip later
# sessions entirely.
SESSIONS = ["ses-0", "ses-1"]

# Only the bipolar pipeline is used (path_utils.get_paths(..., mode='bipolar')),
# so we skip the large monopolar EDFs. Per subject/session we keep:
#   - the bipolar iEEG signal           (*_acq-bipolar_ieeg.edf / .json)
#   - the behavioural events            (beh/*_beh.tsv)
#   - the electrode coordinates         (ieeg/*_electrodes.tsv)
# Set DOWNLOAD_FULL_SUBJECTS = True to instead grab everything (incl. monopolar).
DOWNLOAD_FULL_SUBJECTS = False


def build_includes():
    if DOWNLOAD_FULL_SUBJECTS:
        return [f"{sub}/*" for sub in SUBJECTS]
    includes = []
    for sub in SUBJECTS:
        for ses in SESSIONS:
            includes += [
                f"{sub}/{ses}/ieeg/*acq-bipolar_ieeg*",
                f"{sub}/{ses}/ieeg/*_electrodes.tsv",
                f"{sub}/{ses}/ieeg/*_channels.tsv",
                f"{sub}/{ses}/beh/*_beh.tsv",
            ]
    return includes


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    includes = build_includes()
    print(f"Downloading {len(SUBJECTS)} subjects from OpenNeuro {DATASET}")
    print(f"  -> {TARGET_DIR}")
    print(f"  mode: {'FULL subjects' if DOWNLOAD_FULL_SUBJECTS else 'bipolar + beh + electrodes only'}")
    openneuro.download(
        dataset=DATASET,
        tag=TAG,
        target_dir=TARGET_DIR,
        include=includes,
    )
    print("Done.")


if __name__ == "__main__":
    main()