import numpy as np
import os
import glob
import copy

from paths_and_constants import *


def get_paths(subject, sess_slct=None, mode='monopolar'):

    assert mode in ['monopolar', 'bipolar']
    paths = []
    sessions = glob.glob(os.path.join(BASE_FOLDER, subject, 'ses-*'))
    sessions = [os.path.basename(s) for s in sessions]
    for sess in sessions:
        if (sess_slct is None) or (int(sess[-1]) in sess_slct):
            base_name = '{}_{}_task-FR1'.format(subject, sess)
            electrode_name = base_name + '_space-MNI152NLin6ASym_electrodes.tsv'
            electrode_name = os.path.join(BASE_FOLDER, subject, sess, 'ieeg', electrode_name)
            # if not os.path.isfile(electrode_name):
            #     electrode_name = base_name + '_space-Talairach_electrodes.tsv'
            #     electrode_name = os.path.join(base_folder, subject, sess, 'ieeg', electrode_name)
            event_name = os.path.join(BASE_FOLDER, subject, sess, 'beh', base_name + '_beh.tsv')
            #signal_name = os.path.join(base_folder, subject, sess, 'ieeg', base_name + '_acq-monopolar_ieeg.edf')
            signal_name = os.path.join(BASE_FOLDER, subject, sess, 'ieeg', base_name + '_acq-{}_ieeg.edf'.format(mode))
            if os.path.isfile(electrode_name) and os.path.isfile(event_name) and os.path.isfile(signal_name):
                paths.append(dict({'electrodes': electrode_name, 'events': event_name, 'signals': signal_name}))

    return paths


def get_subject_list():

    subjects = glob.glob(os.path.join(BASE_FOLDER, 'sub-*'))
    subjects = [os.path.basename(s) for s in subjects]

    return subjects



def target_file_name(edf_fname, type, proc_type=None, event_type=None, create_folder=False):

    #
    edf_fname = copy.copy(edf_fname)
    fname_parts = []
    while not (edf_fname == BASE_FOLDER):
        edf_fname, part = os.path.split(edf_fname)
        fname_parts.append(part)
        #print(edf_fname, '\t', part)
    fname_parts = fname_parts[::-1]

    if type == 'annot':
        tgt_fname = os.path.join(PROC_FOLDER, fname_parts[0], fname_parts[1], fname_parts[3].replace('.edf', '.annot'))

    # if type == 'processed':
    #     assert extra_info is not None
    #     return os.path.join(PROC_FOLDER, fname_parts[0], fname_parts[1], fname_parts[2], fname_parts[3].replace('.edf', '_' + extra_info + '_raw.fif'))

    if type == 'processed':
        assert proc_type is not None
        tgt_fname = os.path.join(PROC_FOLDER, fname_parts[0], fname_parts[1], proc_type, fname_parts[3].replace('.edf', '_raw.fif'))

    if type == 'epoched':
        assert proc_type is not None
        assert event_type is not None
        tgt_fname = os.path.join(PROC_FOLDER, fname_parts[0], fname_parts[1], proc_type, fname_parts[3].replace('.edf', '_{}_epo.fif'.format(event_type)))

    if create_folder:
        os.makedirs(os.path.dirname(tgt_fname), exist_ok=True)

    return tgt_fname


