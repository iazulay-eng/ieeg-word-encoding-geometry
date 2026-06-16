import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import argparse
import scipy
import pickle
import logging

from paths_and_constants import *
#from path_utils import get_paths, get_subject_list
import path_utils
from my_mne_wrapper import my_mne_wrapper



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--proc_type', type=str, default='gamma_c_60_160', help='type of response')
    parser.add_argument('--event_type', type=str, default='CNTDWN', help='event type to epoch')
    parser.add_argument('--partition-id', type=int, default=0, help='The ID of the partition to process (0-indexed).')
    parser.add_argument('--num-partitions', type=int, default=1, help='The total number of partitions.')

    args = parser.parse_args()

    logging.basicConfig(filename=os.path.join(LOG_FOLDER, os.path.basename(__file__).replace('.py', '_{}.log'.format(args.partition_id))), filemode='w', level=logging.DEBUG)

    subject_list = path_utils.get_subject_list()
    subject_list = np.sort(subject_list)[args.partition_id::args.num_partitions]

    if args.event_type == 'CNTDWN':
        sub_description = 'DIGIT'
        tmin, tmax = -5, 15.5
    if args.event_type == 'ORIENT':
        tmin, tmax = -2.5-2, 9 + 2.5+2
    if args.event_type == 'LIST':
        sub_description = 'WORD'
        tmin, tmax = -2.5, 30.5 + 2.5
    if args.event_type == 'RECALL':
        tmin, tmax = -2.5, 9 + 2.5
    if args.event_type == 'DSTRCT':
        tmin, tmax = -2.5, 9 + 2.5
    if args.event_type == 'REST':
        tmin, tmax = -2.5, 9 + 2.5

    FORCE_OVERRIDE = False
    for subject in subject_list:
        logging.info('working on {}'.format(subject))
        paths = path_utils.get_paths(subject, mode='bipolar')
        for path in paths:
            edf_name = path['signals']
            raw_name = path_utils.target_file_name(edf_fname=edf_name, type='processed', proc_type=args.proc_type)
            epo_name = path_utils.target_file_name(edf_fname=edf_name, type='epoched', proc_type=args.proc_type, event_type=args.event_type)
            #print(raw_name, ' ==> ', epo_name)

            # Skip if the epoch file already exists (resumable; preserves
            # externally-provided files). Set FORCE_OVERRIDE=True to regenerate.
            if (not FORCE_OVERRIDE) and os.path.isfile(epo_name):
                continue

            try:
                mne_raw = mne.io.read_raw_fif(raw_name, verbose=False)
                # make the event list
                events_for_epoching , src_event_id= mne.events_from_annotations(mne_raw, verbose=False)
                src_event_id = src_event_id[args.event_type]
                events_for_epoching = events_for_epoching[events_for_epoching[:, 2] == src_event_id]
                events_for_epoching[:, 2] = EVENT_TYPES[args.event_type]

                epoched = mne.Epochs(mne_raw, baseline=None, events=events_for_epoching, tmin=tmin, tmax=tmax, reject_by_annotation=False, verbose=False)
                # epoched.plot()
                # plt.show()
                epoched.save(epo_name, overwrite=True, verbose=False)
                logging.info('{}, {} epochs,      {} bad channels'.format(epo_name, events_for_epoching.shape[0], len(epoched.info['bads'])))
                print('{}, {} epochs,      {} bad channels'.format(epo_name, events_for_epoching.shape[0], len(epoched.info['bads'])))
            except:
                logging.warning('FAIL to generate {}'.format(epo_name))
                print('FAIL to generate {}'.format(epo_name))
