"""
Interactive electrode brain maps (Notebook 1, Step 13).

Thin orchestration wrapper around the AAL-labeling + nilearn plotting pipeline
in ``_electrode_brain_map_impl``. Produces one interactive HTML map per subject
(electrodes colored by AAL region) plus a shared legend, and labels each
electrode CSV with ``AAL_Label`` + ``Broad_ROI`` (the tags Step 4 uses).
"""

from . import _electrode_brain_map_impl as _impl


def generate_brain_maps(subjects, paths, out_dir, session="ses-0"):
    """
    Build interactive brain maps for all subjects.

    Parameters
    ----------
    subjects : list of (subject, session) tuples (sessions are de-duplicated).
    paths    : lib.config.Paths (uses paths.raw_dir for the BIDS electrode .tsv).
    out_dir  : directory for the HTML maps + legend PNG.
    session  : which session's electrode coordinates to read (shared across sessions).

    Outputs
    -------
    <out_dir>/<subject>_brain_map.html, <out_dir>/brain_plot_legend.png ;
    updates each <subject>_electrode_locations.csv with AAL_Label + Broad_ROI.
    """
    subs = sorted({s for s, _ in subjects})
    bids_root = str(paths.raw_dir)
    out_dir = str(out_dir)

    aal_data, inv_affine, id_to_label = _impl.load_aal_atlas()
    _impl.step1_export_csvs(subs, bids_root, session, _impl.FILE_TEMPLATE)
    _impl.step2_label_aal(subs, bids_root, aal_data, inv_affine, id_to_label)
    _impl.step2b_label_broad_roi(subs, bids_root)
    color_map = _impl.step3_build_color_map(subs, bids_root)
    _impl.step4_generate_html_plots(subs, bids_root, out_dir, color_map)
    _impl.step5_generate_legend(color_map, out_dir)
