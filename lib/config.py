"""
Configuration for the analysis layer: path resolution + the canonical subject list.

Paths are resolved from the environment so the notebooks run on any machine and,
in particular, can run from the *processed* data tier without re-doing preprocessing:

    1. environment variable  IEEG_DATA_ROOT
    2. a one-line text file   <repo_root>/.ieeg_data_root
    3. fall back to the repo root itself
"""

import os
import pathlib
from dataclasses import dataclass

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    repo_root: pathlib.Path
    data_root: pathlib.Path
    raw_dir: pathlib.Path        # ds004789-download (raw BIDS)
    processed_dir: pathlib.Path  # dr-processed (matrices, epochs, annot)
    results_dir: pathlib.Path    # results/


def _resolve_data_root() -> pathlib.Path:
    root = os.environ.get("IEEG_DATA_ROOT")
    if not root:
        cfg = _REPO_ROOT / ".ieeg_data_root"
        if cfg.is_file():
            root = cfg.read_text(encoding="utf-8").strip()
    return pathlib.Path(root) if root else _REPO_ROOT


def resolve_paths() -> Paths:
    """Resolve all directories used by the notebooks. Creates results/ if missing."""
    data_root = _resolve_data_root()
    results_dir = _REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    return Paths(
        repo_root=_REPO_ROOT,
        data_root=data_root,
        raw_dir=data_root / "ds004789-download",
        processed_dir=data_root / "dr-processed",
        results_dir=results_dir,
    )


# --- Canonical (subject, session) list ---------------------------------------
# 47 subjects x 2 sessions. Two subjects are excluded (file/length problems),
# matching the analysis in the project.
SUBJECTS = [
    ("sub-R1001P", "ses-0"), ("sub-R1001P", "ses-1"),
    ("sub-R1002P", "ses-0"), ("sub-R1002P", "ses-1"),
    ("sub-R1010J", "ses-0"), ("sub-R1010J", "ses-1"),
    ("sub-R1051J", "ses-0"), ("sub-R1051J", "ses-1"),
    ("sub-R1054J", "ses-0"), ("sub-R1054J", "ses-1"),
    ("sub-R1060M", "ses-0"), ("sub-R1060M", "ses-1"),
    ("sub-R1065J", "ses-0"), ("sub-R1065J", "ses-1"),
    ("sub-R1075J", "ses-0"), ("sub-R1075J", "ses-1"),
    ("sub-R1076D", "ses-0"), ("sub-R1076D", "ses-1"),
    ("sub-R1092J", "ses-0"), ("sub-R1092J", "ses-1"),
    ("sub-R1098D", "ses-0"), ("sub-R1098D", "ses-1"),
    ("sub-R1108J", "ses-0"), ("sub-R1108J", "ses-1"),
    ("sub-R1118N", "ses-0"), ("sub-R1118N", "ses-1"),
    ("sub-R1123C", "ses-0"), ("sub-R1123C", "ses-1"),
    ("sub-R1136N", "ses-0"), ("sub-R1136N", "ses-1"),
    ("sub-R1145J", "ses-0"), ("sub-R1145J", "ses-1"),
    ("sub-R1151E", "ses-0"), ("sub-R1151E", "ses-1"),
    ("sub-R1153T", "ses-0"), ("sub-R1153T", "ses-1"),
    ("sub-R1154D", "ses-0"), ("sub-R1154D", "ses-1"),
    ("sub-R1156D", "ses-0"), ("sub-R1156D", "ses-1"),
    ("sub-R1161E", "ses-0"), ("sub-R1161E", "ses-1"),
    ("sub-R1166D", "ses-0"), ("sub-R1166D", "ses-1"),
    ("sub-R1168T", "ses-0"), ("sub-R1168T", "ses-1"),
    ("sub-R1173J", "ses-0"), ("sub-R1173J", "ses-1"),
    ("sub-R1189M", "ses-0"), ("sub-R1189M", "ses-1"),
    ("sub-R1195E", "ses-0"), ("sub-R1195E", "ses-1"),
    ("sub-R1196N", "ses-0"), ("sub-R1196N", "ses-1"),
    ("sub-R1200T", "ses-0"), ("sub-R1200T", "ses-1"),
    ("sub-R1201P", "ses-0"), ("sub-R1201P", "ses-1"),
    ("sub-R1223E", "ses-0"), ("sub-R1223E", "ses-1"),
    ("sub-R1234D", "ses-0"), ("sub-R1234D", "ses-1"),
    ("sub-R1243T", "ses-0"), ("sub-R1243T", "ses-1"),
    ("sub-R1283T", "ses-0"), ("sub-R1283T", "ses-1"),
    ("sub-R1297T", "ses-0"), ("sub-R1297T", "ses-1"),
    ("sub-R1292E", "ses-0"), ("sub-R1292E", "ses-1"),
    ("sub-R1299T", "ses-0"), ("sub-R1299T", "ses-1"),
    ("sub-R1310J", "ses-0"), ("sub-R1310J", "ses-1"),
    ("sub-R1315T", "ses-0"), ("sub-R1315T", "ses-1"),
    ("sub-R1317D", "ses-0"), ("sub-R1317D", "ses-1"),
    ("sub-R1328E", "ses-0"), ("sub-R1328E", "ses-1"),
    ("sub-R1332M", "ses-0"), ("sub-R1332M", "ses-1"),
    ("sub-R1334T", "ses-0"), ("sub-R1334T", "ses-1"),
    ("sub-R1338T", "ses-0"), ("sub-R1338T", "ses-1"),
    ("sub-R1346T", "ses-0"), ("sub-R1346T", "ses-1"),
    ("sub-R1350D", "ses-0"), ("sub-R1350D", "ses-1"),
    ("sub-R1354E", "ses-0"), ("sub-R1354E", "ses-1"),
    ("sub-R1355T", "ses-0"), ("sub-R1355T", "ses-1"),
    ("sub-R1425D", "ses-0"), ("sub-R1425D", "ses-1"),
]
