"""
Centralized paths and constants for the preprocessing pipeline.

Large raw/processed data normally live OUTSIDE the code repository (so a synced
folder does not try to upload tens of GB). The data root is resolved in this order:

    1. environment variable  IEEG_DATA_ROOT
    2. a one-line text file   <repo_root>/.ieeg_data_root
    3. fall back to the repo root itself

Set whichever is convenient, e.g.:
    export IEEG_DATA_ROOT=/path/to/ieeg_data       (Linux/Mac)
    setx    IEEG_DATA_ROOT  D:\\ieeg_data            (Windows)
"""

import os
import pathlib

# Repo root = one level above this file (preprocessing/).
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)

# --- Resolve the data root (no machine-specific paths hard-coded) ------------
DATA_ROOT = os.environ.get("IEEG_DATA_ROOT")
if not DATA_ROOT:
    _cfg = os.path.join(_REPO_ROOT, ".ieeg_data_root")
    if os.path.isfile(_cfg):
        with open(_cfg, "r", encoding="utf-8") as _fd:
            DATA_ROOT = _fd.read().strip()
DATA_ROOT = DATA_ROOT or _REPO_ROOT

IS_CLUSTER = bool(os.environ.get("IEEG_CLUSTER"))   # optional flag; no path assumptions

# --- Folder layout -----------------------------------------------------------
# Big folders live under DATA_ROOT; small bookkeeping folders stay in the repo.
BASE_FOLDER = os.path.join(DATA_ROOT, "ds004789-download")   # raw BIDS dataset
PROC_FOLDER = os.path.join(DATA_ROOT, "dr-processed")        # processed matrices / epochs
TEMP_FOLDER = os.path.join(DATA_ROOT, "dr-temp")             # scratch
LOG_FOLDER = os.path.join(_REPO_ROOT, "logs")
IDXS_FOLDER = os.path.join(_REPO_ROOT, "dr-indexes")
HOME_DIR = os.path.expanduser("~")

# --- Event-type codes used throughout the pipeline ---------------------------
EVENT_TYPES = {
    "CNTDWN": 10, "DIGIT": 11, "LIST": 20, "WORD": 21, "ORIENT": 30,
    "RANDOM": 40, "RECALL": 50, "DSTRCT": 60, "REST": 70,
    10: "CNTDWN", 11: "DIGIT", 20: "LIST", 21: "WORD", 30: "ORIENT",
    40: "RANDOM", 50: "RECALL", 60: "DSTRCT", 70: "REST",
}
