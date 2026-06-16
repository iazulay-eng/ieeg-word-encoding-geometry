"""
Fetch the processed-data bundle (Tier B) so the project runs WITHOUT the raw
dataset and WITHOUT re-doing preprocessing.

The bundle is a ~5 GB tar containing, per session, the four files the analysis
needs:
    dr-processed/<subject>/<session>/
        <sub>_<ses>_aligned_matrix_normratio.npy
        <sub>_<ses>_aligned_channels_raw.pkl
        correlations_scores_normratio.csv
        <sub>_<ses>_task-FR1_acq-bipolar_ieeg.annot

It is hosted on Google Drive. Downloading large Drive files reliably needs
`gdown` (handles the confirm-token), so this script uses it:

    pip install gdown

Usage
-----
    python data/download_data.py                       # uses GDRIVE_FILE_ID below / env
    python data/download_data.py --id <drive_file_id>
    python data/download_data.py --dest D:\\ieeg_data    # override data root
"""

import argparse
import os
import pathlib
import sys
import tarfile

# Google Drive FILE ID of the bundle (the long token in the share link
# https://drive.google.com/file/d/<THIS>/view).
GDRIVE_FILE_ID = "1-WpHqphLHEBA4eC2Yat86eUGia-vsrJ4"


def _resolve_dest():
    root = os.environ.get("IEEG_DATA_ROOT")
    if not root:
        cfg = pathlib.Path(__file__).resolve().parent.parent / ".ieeg_data_root"
        if cfg.is_file():
            root = cfg.read_text(encoding="utf-8").strip()
    return pathlib.Path(root) if root else pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Download the processed-data bundle (Tier B) from Google Drive.")
    ap.add_argument("--id", default=os.environ.get("IEEG_GDRIVE_ID", GDRIVE_FILE_ID),
                    help="Google Drive file id of the bundle tar")
    ap.add_argument("--dest", default=None, help="data root (defaults to IEEG_DATA_ROOT / .ieeg_data_root)")
    args = ap.parse_args()

    if not args.id:
        sys.exit("No Google Drive file id. Pass --id, set IEEG_GDRIVE_ID, or fill GDRIVE_FILE_ID.")

    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required for Google Drive downloads:  pip install gdown")

    dest = pathlib.Path(args.dest) if args.dest else _resolve_dest()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "ieeg-processed-min.tar"

    print(f"Downloading bundle from Google Drive (id={args.id})\n-> {archive}")
    gdown.download(id=args.id, output=str(archive), quiet=False)

    print("Extracting...")
    with tarfile.open(archive) as t:
        t.extractall(dest)            # creates dest/dr-processed/<subject>/<session>/...

    archive.unlink(missing_ok=True)
    print(f"Done. dr-processed/ is ready under {dest}")
    print("Now set IEEG_DATA_ROOT to this folder (or write it into <repo>/.ieeg_data_root).")


if __name__ == "__main__":
    main()
