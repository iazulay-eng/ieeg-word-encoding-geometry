"""
electrode_brain_map.py
======================
Pipeline for labeling iEEG electrode locations with AAL brain regions
and generating per-subject interactive HTML brain maps + a legend PNG.

Pipeline Steps
--------------
  Step 1  Export electrode (x,y,z) coordinates from BIDS .tsv to per-subject CSVs
  Step 2  Load the AAL atlas via nilearn and label each electrode with its brain region
  Step 3  Build a single, consistent color map across all subjects
  Step 4  Generate one interactive HTML brain plot per subject  (ALL electrodes)
  Step 5  Generate a shared color-coded region legend PNG

Usage
-----
  1. Fill in the CONFIGURATION block below.
  2. Run:  python electrode_brain_map.py
"""

# ==============================================================================
# Imports
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from nilearn import datasets, plotting
from nilearn.image import load_img, get_data, coord_transform


# ==============================================================================
# CONFIGURATION  (edit this section)
# ==============================================================================

import pathlib as _pathlib
import glob as _glob
import os as _os

_PROJECT_ROOT = str(_pathlib.Path(__file__).parent.parent)
_HOME = _os.path.expanduser("~")

# Root of the raw BIDS dataset (contains sub-* folders)
BIDS_ROOT = str(_pathlib.Path(_PROJECT_ROOT) / "ds004789-download")

# Directory where HTML brain plots and the legend PNG will be saved
OUTPUT_DIR = str(_pathlib.Path(_PROJECT_ROOT) / "brain-plots")

# Subjects to process — all subjects found in BIDS root
SUBJECTS = sorted([
    _os.path.basename(p)
    for p in _glob.glob(str(_pathlib.Path(BIDS_ROOT) / "sub-*"))
])

# Session to read electrode coordinates from.
# All sessions share the same electrode locations, so one is enough.
SESSION = "ses-0"

# BIDS filename template for the electrode coordinates .tsv file.
# Actual filename on disk: sub-R1001P_ses-0_task-FR1_space-MNI152NLin6ASym_electrodes.tsv
FILE_TEMPLATE = (
    "{subject}_{session}_task-FR1_space-MNI152NLin6ASym_electrodes.tsv"
)

# Local path to the AAL atlas NII file (ROI_MNI_V4.nii).
ATLAS_NII = str(_pathlib.Path(_HOME) / "nilearn_data" / "aal_SPM12" / "ROI_MNI_V4.nii")
ATLAS_TXT = str(_pathlib.Path(_HOME) / "nilearn_data" / "aal_SPM12" / "ROI_MNI_V4.txt")

# Output file names
LEGEND_FILENAME = "brain_plot_legend.png"


# ==============================================================================
# AAL Full-Name Lookup
# ==============================================================================
# Maps the base part of a standard AAL label (without _L / _R suffix)
# to a human-readable full name.  Covers all regions in AAL-SPM12/3v2.

_AAL_FULL_NAMES: dict[str, str] = {
    # Frontal
    "Frontal_Sup":          "Superior Frontal Gyrus",
    "Frontal_Sup_Orb":      "Superior Frontal Gyrus (Orbital)",
    "Frontal_Mid":          "Middle Frontal Gyrus",
    "Frontal_Mid_Orb":      "Middle Frontal Gyrus (Orbital)",
    "Frontal_Inf_Oper":     "Inferior Frontal Gyrus (Opercular)",
    "Frontal_Inf_Tri":      "Inferior Frontal Gyrus (Triangular)",
    "Frontal_Inf_Orb":      "Inferior Frontal Gyrus (Orbital)",
    "Rolandic_Oper":        "Rolandic Operculum",
    "Supp_Motor_Area":      "Supplementary Motor Area",
    "Olfactory":            "Olfactory Cortex",
    "Frontal_Sup_Medial":   "Superior Frontal Gyrus (Medial)",
    "Frontal_Med_Orb":      "Medial Frontal Gyrus (Orbital)",
    "Rectus":               "Gyrus Rectus",
    "Precentral":           "Precentral Gyrus",
    # Parietal
    "Postcentral":          "Postcentral Gyrus",
    "Parietal_Sup":         "Superior Parietal Gyrus",
    "Parietal_Inf":         "Inferior Parietal Gyrus",
    "SupraMarginal":        "Supramarginal Gyrus",
    "Angular":              "Angular Gyrus",
    "Precuneus":            "Precuneus",
    "Paracentral_Lobule":   "Paracentral Lobule",
    # Occipital
    "Calcarine":            "Calcarine Fissure",
    "Cuneus":               "Cuneus",
    "Lingual":              "Lingual Gyrus",
    "Occipital_Sup":        "Superior Occipital Gyrus",
    "Occipital_Mid":        "Middle Occipital Gyrus",
    "Occipital_Inf":        "Inferior Occipital Gyrus",
    "Fusiform":             "Fusiform Gyrus",
    # Temporal
    "Temporal_Sup":         "Superior Temporal Gyrus",
    "Temporal_Pole_Sup":    "Temporal Pole (Superior)",
    "Temporal_Mid":         "Middle Temporal Gyrus",
    "Temporal_Pole_Mid":    "Temporal Pole (Middle)",
    "Temporal_Inf":         "Inferior Temporal Gyrus",
    "Heschl":               "Heschl Gyrus",
    # MTL
    "Hippocampus":          "Hippocampus",
    "ParaHippocampal":      "Parahippocampal Gyrus",
    "Amygdala":             "Amygdala",
    # Cingulate
    "Cingulum_Ant":         "Anterior Cingulate Cortex",
    "Cingulum_Mid":         "Middle Cingulate Cortex",
    "Cingulum_Post":        "Posterior Cingulate Cortex",
    # Other cortical
    "Insula":               "Insula",
    # Subcortical
    "Thalamus":             "Thalamus",
    "Caudate":              "Caudate Nucleus",
    "Putamen":              "Putamen",
    "Pallidum":             "Pallidum",
    "Cerebelum_Crus1":      "Cerebellum Crus 1",
    "Cerebelum_Crus2":      "Cerebellum Crus 2",
    "Cerebelum_3":          "Cerebellum Lobule 3",
    "Cerebelum_4_5":        "Cerebellum Lobules 4-5",
    "Cerebelum_6":          "Cerebellum Lobule 6",
    "Cerebelum_7b":         "Cerebellum Lobule 7b",
    "Cerebelum_8":          "Cerebellum Lobule 8",
    "Cerebelum_9":          "Cerebellum Lobule 9",
    "Cerebelum_10":         "Cerebellum Lobule 10",
    "Vermis_1_2":           "Cerebellar Vermis 1-2",
    "Vermis_3":             "Cerebellar Vermis 3",
    "Vermis_4_5":           "Cerebellar Vermis 4-5",
    "Vermis_6":             "Cerebellar Vermis 6",
    "Vermis_7":             "Cerebellar Vermis 7",
    "Vermis_8":             "Cerebellar Vermis 8",
    "Vermis_9":             "Cerebellar Vermis 9",
    "Vermis_10":            "Cerebellar Vermis 10",
}


# ==============================================================================
# Broad ROI Mapping  (5 regions)
# ==============================================================================

_BROAD_ROI_MAPPING: dict[str, str] = {
    # IFG
    "Inferior Frontal Gyrus (Opercular)":  "IFG",
    "Inferior Frontal Gyrus (Triangular)": "IFG",
    "Inferior Frontal Gyrus (Orbital)":    "IFG",
    # PFC
    "Superior Frontal Gyrus":              "PFC",
    "Superior Frontal Gyrus (Orbital)":    "PFC",
    "Middle Frontal Gyrus":                "PFC",
    "Middle Frontal Gyrus (Orbital)":      "PFC",
    "Superior Frontal Gyrus (Medial)":     "PFC",
    "Medial Frontal Gyrus (Orbital)":      "PFC",
    "Gyrus Rectus":                        "PFC",
    "Olfactory Cortex":                    "PFC",
    # MTL
    "Hippocampus":                         "MTL",
    "Parahippocampal Gyrus":               "MTL",
    "Amygdala":                            "MTL",
    # Temporal
    "Superior Temporal Gyrus":             "Temporal",
    "Temporal Pole (Superior)":            "Temporal",
    "Middle Temporal Gyrus":               "Temporal",
    "Temporal Pole (Middle)":              "Temporal",
    "Inferior Temporal Gyrus":             "Temporal",
    "Heschl Gyrus":                        "Temporal",
    # Occipital
    "Calcarine Fissure":                   "Occipital",
    "Cuneus":                              "Occipital",
    "Lingual Gyrus":                       "Occipital",
    "Superior Occipital Gyrus":            "Occipital",
    "Middle Occipital Gyrus":              "Occipital",
    "Inferior Occipital Gyrus":            "Occipital",
    "Fusiform Gyrus":                      "Occipital",
}


def format_aal_label(raw_label: str) -> str:
    """
    Convert a raw AAL label string to a readable display name.

    Examples
    --------
    'Frontal_Mid_L'        -> 'Middle Frontal Gyrus (Left)'
    'Hippocampus_R'        -> 'Hippocampus (Right)'
    'Outside_AAL'          -> 'Outside Atlas'
    'Temporal_Sup_L'       -> 'Superior Temporal Gyrus (Left)'
    """
    raw_label = str(raw_label).strip()

    if raw_label in ("Outside_AAL", "Unknown", ""):
        return "Outside Atlas"

    side = ""
    base = raw_label

    if raw_label.endswith("_L"):
        side = " (Left)"
        base = raw_label[:-2]
    elif raw_label.endswith("_R"):
        side = " (Right)"
        base = raw_label[:-2]

    full_name = _AAL_FULL_NAMES.get(base, base.replace("_", " "))
    return f"{full_name}{side}"


def assign_broad_roi(aal_label: str) -> str:
    """
    Map a raw AAL label (e.g. 'Frontal_Inf_Tri_L') to one of 5 broad ROIs:
    IFG, PFC, MTL, Temporal, Occipital.  Returns 'Other' for everything else.
    """
    readable = format_aal_label(aal_label)          # e.g. 'Inferior Frontal Gyrus (Triangular) (Left)'
    base = readable.replace(" (Left)", "").replace(" (Right)", "").strip()
    return _BROAD_ROI_MAPPING.get(base, "Other")


# ==============================================================================
# Atlas Loading
# ==============================================================================

def load_aal_atlas(nii_path: str = ATLAS_NII, txt_path: str = ATLAS_TXT) -> tuple:
    """
    Load the AAL atlas directly from local NII and TXT files.

    This avoids any network download. The files are the standard AAL-SPM12
    distribution (ROI_MNI_V4.nii + ROI_MNI_V4.txt).

    TXT file format (3 whitespace-separated columns, no header):
        <code>  <label>  <region_id>
    Example:
        FAG  Precentral_L  2001
        FAD  Precentral_R  2002

    Returns
    -------
    aal_data : np.ndarray
        3-D integer array of the atlas image (voxel parcellation).
    inv_affine : np.ndarray
        4x4 inverse affine matrix for MNI-mm -> voxel conversion.
    id_to_label : dict[int, str]
        Mapping from integer region ID (as stored in aal_data) to
        the AAL label string (e.g. 'Frontal_Mid_L').
    """
    print("Loading AAL atlas from local files...")
    print("  NII: {}".format(nii_path))
    print("  TXT: {}".format(txt_path))

    aal_img    = load_img(nii_path)
    aal_data   = get_data(aal_img)
    affine     = aal_img.affine
    inv_affine = np.linalg.inv(affine)

    # Parse TXT: columns are  code | label | region_id
    labels_df = pd.read_csv(
        txt_path, sep=r"\s+", header=None,
        names=["code", "label", "region_id"],
    )
    id_to_label: dict[int, str] = dict(
        zip(labels_df["region_id"].astype(int), labels_df["label"])
    )

    print("  Atlas loaded: {} regions defined.".format(len(id_to_label)))
    return aal_data, inv_affine, id_to_label


# ==============================================================================
# Step 1 — Export Electrode Coordinates to CSV
# ==============================================================================

def step1_export_csvs(
    subjects:      list[str],
    bids_root:     str,
    session:       str,
    file_template: str,
) -> None:
    """
    Read each subject's BIDS electrode .tsv file and save a clean CSV
    containing only [name, x, y, z] columns.

    Output
    ------
    <bids_root>/<subject>/<subject>_electrode_locations.csv
    """
    print("\n" + "=" * 60)
    print("Step 1 — Exporting electrode coordinates to CSV")
    print("=" * 60)

    required_columns = ["name", "x", "y", "z"]

    for subject in subjects:
        print(f"\n  [{subject}]")

        file_name = file_template.format(subject=subject, session=session)
        file_path = os.path.join(bids_root, subject, session, "ieeg", file_name)

        try:
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            print(f"    SKIP: TSV not found at {file_path}")
            continue
        except Exception as exc:
            print(f"    ERROR reading file: {exc}")
            continue

        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            print(f"    SKIP: Missing columns {missing}")
            continue

        df_clean = df[required_columns].copy()

        output_path = os.path.join(bids_root, subject,
                                   f"{subject}_electrode_locations.csv")
        df_clean.to_csv(output_path, index=False)

        print(f"    OK: {len(df_clean)} electrodes -> {output_path}")

    print("\nStep 1 complete.")


# ==============================================================================
# Step 2 — Label Electrodes with AAL Regions
# ==============================================================================

def step2_label_aal(
    subjects:     list[str],
    bids_root:    str,
    aal_data:     np.ndarray,
    inv_affine:   np.ndarray,
    id_to_label:  dict[int, str],
) -> None:
    """
    For each subject, add an 'AAL_Label' column to the electrode CSV
    by converting MNI coordinates to atlas voxel indices and looking up
    the region label.

    Updates each CSV in-place.
    """
    print("\n" + "=" * 60)
    print("Step 2 — Labeling electrodes with AAL regions")
    print("=" * 60)

    for subject in subjects:
        print(f"\n  [{subject}]")

        csv_path = os.path.join(bids_root, subject,
                                f"{subject}_electrode_locations.csv")
        if not os.path.exists(csv_path):
            print(f"    SKIP: CSV not found — run Step 1 first.")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"    ERROR reading CSV: {exc}")
            continue

        aal_labels = []

        for _, row in df.iterrows():
            # Convert MNI (mm) -> voxel indices using the inverse affine
            vox_coords = coord_transform(
                row["x"], row["y"], row["z"], inv_affine
            )
            vi, vj, vk = np.round(vox_coords).astype(int)

            label = "Outside_AAL"

            if (0 <= vi < aal_data.shape[0] and
                    0 <= vj < aal_data.shape[1] and
                    0 <= vk < aal_data.shape[2]):

                region_id = int(aal_data[vi, vj, vk])
                if region_id in id_to_label:
                    label = id_to_label[region_id]

            aal_labels.append(label)

        df["AAL_Label"] = aal_labels
        df.to_csv(csv_path, index=False)

        # Brief region summary for the first electrode
        region_counts = df["AAL_Label"].value_counts()
        print(f"    OK: {len(df)} electrodes labeled across "
              f"{len(region_counts)} regions.")
        print(f"    Top region: {region_counts.index[0]} "
              f"({region_counts.iloc[0]} electrodes)")

    print("\nStep 2 complete.")


# ==============================================================================
# Step 2b — Add Broad ROI Column
# ==============================================================================

def step2b_label_broad_roi(
    subjects:  list[str],
    bids_root: str,
) -> None:
    """
    Add a 'Broad_ROI' column to each subject's electrode CSV.
    Maps each AAL_Label to one of: IFG, PFC, MTL, Temporal, Occipital, Other.
    Requires step2_label_aal to have run first (AAL_Label column must exist).
    """
    print("\n" + "=" * 60)
    print("Step 2b — Broad ROI labeling")

    for subject in subjects:
        csv_path = os.path.join(bids_root, subject, f"{subject}_electrode_locations.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {subject}: CSV not found")
            continue

        try:
            df = pd.read_csv(csv_path)
            if "AAL_Label" not in df.columns:
                print(f"  [SKIP] {subject}: AAL_Label column missing — run step2 first")
                continue

            df["Broad_ROI"] = df["AAL_Label"].apply(assign_broad_roi)
            df.to_csv(csv_path, index=False)

            print(f"  OK: {subject}")
            print(df["Broad_ROI"].value_counts().to_string())
        except Exception as exc:
            print(f"  [ERROR] {subject}: {exc}")

    print("\nStep 2b complete.")


# ==============================================================================
# Step 3 — Build Global Color Map
# ==============================================================================

def step3_build_color_map(
    subjects:  list[str],
    bids_root: str,
) -> dict[str, str]:
    """
    Collect every unique AAL region that appears across all subjects,
    then assign a consistent hex color to each one.

    A single color map is built here so that the same region has the same
    color in every subject's HTML plot and in the legend.

    Returns
    -------
    dict mapping region label (str) -> hex color string (str)
    """
    print("\n" + "=" * 60)
    print("Step 3 — Building global color map")
    print("=" * 60)

    all_regions: set[str] = set()

    for subject in subjects:
        csv_path = os.path.join(bids_root, subject,
                                f"{subject}_electrode_locations.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "AAL_Label" in df.columns:
            all_regions.update(df["AAL_Label"].dropna().unique())

    unique_regions = sorted(all_regions)
    n = len(unique_regions)

    cmap = plt.get_cmap("tab20") if n <= 20 else plt.get_cmap("nipy_spectral")

    color_map: dict[str, str] = {}
    for i, region in enumerate(unique_regions):
        rgba = cmap(i / n) if n > 1 else cmap(0.0)
        color_map[region] = mcolors.to_hex(rgba)

    print(f"  {n} unique regions found across all subjects.")
    return color_map


# ==============================================================================
# Step 4 — Generate Per-Subject HTML Brain Plots
# ==============================================================================

def step4_generate_html_plots(
    subjects:   list[str],
    bids_root:  str,
    output_dir: str,
    color_map:  dict[str, str],
) -> None:
    """
    For each subject, generate one interactive HTML brain plot showing
    ALL electrodes, colored by their AAL region.

    Output
    ------
    <output_dir>/<subject>_brain_map.html
    """
    print("\n" + "=" * 60)
    print("Step 4 — Generating per-subject HTML brain plots")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    for subject in subjects:
        print(f"\n  [{subject}]")

        csv_path = os.path.join(bids_root, subject,
                                f"{subject}_electrode_locations.csv")
        if not os.path.exists(csv_path):
            print(f"    SKIP: CSV not found — run Steps 1 and 2 first.")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"    ERROR reading CSV: {exc}")
            continue

        # Drop rows with missing coordinates or labels
        df = df.dropna(subset=["x", "y", "z"])

        if df.empty:
            print("    SKIP: No valid electrode coordinates.")
            continue

        # Assign region and color for each electrode
        def _region(row) -> str:
            if "AAL_Label" in df.columns and pd.notna(row.get("AAL_Label")):
                return str(row["AAL_Label"])
            return "Outside_AAL"

        df["_region"] = df.apply(_region, axis=1)
        df["_color"]  = df["_region"].map(
            lambda r: color_map.get(r, "#888888")
        )

        # Build label strings for hover tooltips — electrode name only
        df["_label"] = df["name"].astype(str)

        coords = df[["x", "y", "z"]].values
        colors = df["_color"].tolist()
        labels = df["_label"].tolist()

        view = plotting.view_markers(
            coords,
            marker_labels=labels,
            marker_size=6.0,
            marker_color=colors,
            title=f"{subject} — Electrodes by AAL Region",
        )

        html_path = os.path.join(output_dir, f"{subject}_brain_map.html")
        view.save_as_html(html_path)

        print(f"    OK: {len(df)} electrodes plotted -> {html_path}")

    print("\nStep 4 complete.")


# ==============================================================================
# Step 5 — Generate Legend PNG
# ==============================================================================

def step5_generate_legend(
    color_map:  dict[str, str],
    output_dir: str,
    filename:   str = LEGEND_FILENAME,
) -> None:
    """
    Save a PNG image showing each AAL region label alongside its color swatch.
    Regions are sorted alphabetically; full human-readable names are shown
    next to each color patch.

    Output
    ------
    <output_dir>/<filename>
    """
    print("\n" + "=" * 60)
    print("Step 5 — Generating legend PNG")
    print("=" * 60)

    if not color_map:
        print("  SKIP: Color map is empty — no regions to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    unique_regions = sorted(color_map.keys())

    fig_height = max(2.0, len(unique_regions) * 0.35 + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.axis("off")

    patches = [
        mpatches.Patch(
            color=color_map[region],
            label=format_aal_label(region),
        )
        for region in unique_regions
    ]

    legend = ax.legend(
        handles=patches,
        loc="center",
        frameon=False,
        title="Anatomical Regions (AAL)",
        fontsize=10,
        title_fontsize=12,
        labelspacing=0.8,
    )

    for text in legend.get_texts():
        text.set_ha("left")

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"  OK: {len(unique_regions)} regions -> {save_path}")
    print("\nStep 5 complete.")


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    print("=" * 60)
    print("  iEEG Electrode Brain Map Pipeline")
    print("=" * 60)
    print(f"  BIDS root  : {BIDS_ROOT}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Subjects   : {SUBJECTS}")
    print(f"  Session    : {SESSION}")
    print(f"  Atlas NII  : {ATLAS_NII}")

    # Load atlas once — shared across all steps
    aal_data, inv_affine, id_to_label = load_aal_atlas()

    # Run the pipeline
    step1_export_csvs(
        subjects=SUBJECTS,
        bids_root=BIDS_ROOT,
        session=SESSION,
        file_template=FILE_TEMPLATE,
    )

    step2_label_aal(
        subjects=SUBJECTS,
        bids_root=BIDS_ROOT,
        aal_data=aal_data,
        inv_affine=inv_affine,
        id_to_label=id_to_label,
    )

    step2b_label_broad_roi(
        subjects=SUBJECTS,
        bids_root=BIDS_ROOT,
    )

    color_map = step3_build_color_map(
        subjects=SUBJECTS,
        bids_root=BIDS_ROOT,
    )

    step4_generate_html_plots(
        subjects=SUBJECTS,
        bids_root=BIDS_ROOT,
        output_dir=OUTPUT_DIR,
        color_map=color_map,
    )

    step5_generate_legend(
        color_map=color_map,
        output_dir=OUTPUT_DIR,
    )

    print("\n" + "=" * 60)
    print("  Pipeline finished.")
    print(f"  HTML plots and legend saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
