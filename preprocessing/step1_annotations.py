import copy
import os
import mne.preprocessing
import numpy as np
import matplotlib.pyplot as plt
import path_utils
import pickle
#import json
import argparse
from tqdm import tqdm

#from my_montage_reader import my_montage_reader
from my_mne_wrapper import my_mne_wrapper
from event_reader import event_reader

from paths_and_constants import *
from noise_classifier import noise_classifier

import logging
#logging.basicConfig(filename=os.path.join(ASE_BFOLDER, os.path.basename(__file__).replace('.py', '.log')), filemode='w', level=logging.DEBUG)



#subject_list =  path_utils.get_subject_list()


def generate_epoched_version(path, mne_copy, event_obj, create_new=True, event_name='cntdwn'):

    epoched_fname = path['signals'].replace(BASE_FOLDER, PROC_FOLDER).replace('ieeg', event_name, 1).replace('.edf', '-epo.fif')
    if os.path.isfile(epoched_fname) and (not create_new):
        logging.info(epoched_fname + 'ALREADY EXISTS')
        return

    fs = mne_copy.info['sfreq']

    if event_name == 'random':
        epoching_events = event_obj.get_countdowns()
        # RANDOMIZE ONSETS
        rand_samps = np.sort(np.random.uniform(low=1000, high=mne_copy.times.size-20000, size=len(epoching_events))).astype(int)
        rand_onset = mne_copy.times[rand_samps]
        for i_event, event in enumerate(epoching_events):
            event['onset'] = rand_onset[i_event]
            event['onset sample'] = rand_samps[i_event]
            event['end sample'] = rand_samps[i_event] + 5250
        #
        description = 'RANDOM'
        sub_description = None
        tmin, tmax = -2.5, 10.5 + 2.5
        sub_description, sub_events = None, None
    if event_name == 'cntdwn':
        epoching_events = event_obj.get_countdowns()
        description = 'CNTDWN'
        sub_description = 'DIGIT'
        tmin, tmax = -2.5-2.5, 10.5 + 2.5+2.5
        # generating the DIGIT annotations
        sub_events = []
        for cntdwn_event in epoching_events:
            sub_events = sub_events + [{'onset':  cntdwn_event['onset'] + float(i), 'end': cntdwn_event['onset'] + float(i + 1),
                                        'onset sample': cntdwn_event['onset sample'] + int(fs * i),
                                        'end sample': cntdwn_event['onset sample'] + int(fs * (i + 1))}  for i in range(10)]
    if event_name == 'orient':
        epoching_events = event_obj.get_orients()
        description = 'ORIENT'
        sub_description, sub_events = None, None
        tmin, tmax = -2.5-2, 9 + 2.5+2
    if event_name == 'list':
        epoching_events = event_obj.get_list_events()
        description = 'LIST'
        sub_events = event_obj.get_word_events()
        sub_description = 'WORD'
        tmin, tmax = -2.5, 30.5 + 2.5
    if event_name == 'recall':
        epoching_events = event_obj.get_recalls(random_start=True, max_duration=12)
        description = 'RECALL'
        sub_description, sub_events = None, None
        tmin, tmax = -2.5, 9 + 2.5
    if event_name == 'dstrct':
        epoching_events = event_obj.get_distracts(random_start=True, max_duration=12)
        description = 'DSTRCT'
        sub_description, sub_events = None, None
        tmin, tmax = -2.5, 9 + 2.5
    if event_name == 'rest':
        epoching_events = event_obj.get_rests(random_start=True, max_duration=12)
        description = 'REST'
        sub_description, sub_events = None, None
        tmin, tmax = -2.5, 9 + 2.5

    event_type = EVENT_TYPES[description]
    sub_event_type = None if sub_description == None else EVENT_TYPES[sub_description]

    #onset = np.array([e['onset'] for e in countdown_events])
    onset = np.array([e['onset sample'] / fs for e in epoching_events])
    duration = np.array([(e['end sample'] - e['onset sample']) / fs for e in epoching_events])
    mne_copy.annotations.append(onset, duration, description)
    if sub_events is not None:
        sub_onset = np.array([e['onset sample'] / fs for e in sub_events])
        sub_duration = np.array([(e['end sample'] - e['onset sample']) / fs for e in sub_events])
        mne_copy.annotations.append(sub_onset, sub_duration, sub_description)
    onset_sample = np.array([e['onset sample'] for e in epoching_events])
    events_for_epoching = np.zeros((onset_sample.size, 3), dtype=int)
    events_for_epoching[:, 0] = onset_sample
    events_for_epoching[:, 2] = event_type
    # crop_start = onset - 2
    # crop_end = onset + duration + 2
    # make sure the events are chronologically ordered
    order = np.argsort(events_for_epoching[:, 0])
    events_for_epoching = events_for_epoching[order]

    # save
    #mne_copy._data[2] = 1e-6 * (np.arange(mne_copy._data.shape[-1]) % 1000 - 500)
    epoched = mne.Epochs(mne_copy, events=events_for_epoching, tmin=tmin, tmax=tmax, reject_by_annotation=False, verbose=False)
    #epoched.ch_amps = mne_copy.get_data().std(axis=1)
    os.makedirs(os.path.dirname(epoched_fname), exist_ok=True)
    epoched.save(epoched_fname, overwrite=True, verbose=False)

    SHOW = False
    if SHOW:
        epoched1 = mne.read_epochs(epoched_fname)
        _ = epoched1.plot(scalings=2e-4)
        plt.show()



def check_if_files_exist(src_path, event_names, force_override, verbose=True):

    exist_mask = np.zeros(len(event_names), dtype=bool)
    if not force_override:
        for i, event_name in enumerate(event_names):
            epoched_path = src_path['signals'].replace(BASE_FOLDER, PROC_FOLDER).replace('ieeg', event_name, 1).replace('.edf', '-epo.fif')
            exist_mask[i] = os.path.isfile(epoched_path)
            if exist_mask[i] and verbose:
                logging.info(epoched_path + '  ALREADY EXISTS')

    events_to_process = [event_names[i] for i in np.argwhere(np.logical_not(exist_mask)).flatten()]
    return exist_mask, events_to_process


def add_annotations(mne_obj, events, description):

    for event in events:
        if 'end' in list(event.keys()):
            #mne_obj.annotations.append(event['onset'], event['end'] - event['onset'], description)
            mne_obj.annotations.append(event['onset sample'] / mne_obj.info['sfreq'], event['end'] - event['onset'], description)
        elif 'duration' in list(event.keys()):
            #mne_obj.annotations.append(event['onset'], event['duration'], description)
            mne_obj.annotations.append(event['onset sample'] / mne_obj.info['sfreq'], event['duration'], description)
        # if event['interim events']:
        #     print('here')





if __name__ == '__main__':
    
    # Parse command-line arguments for partitioning
    parser = argparse.ArgumentParser(description='Process data folders in parallel.')
    parser.add_argument('--partition-id', type=int, default=0, help='The ID of the partition to process (0-indexed).')
    parser.add_argument('--num-partitions', type=int, default=1, help='The total number of partitions.')
    
    args = parser.parse_args()
    

    logging.basicConfig(filename=os.path.join(LOG_FOLDER, os.path.basename(__file__).replace('.py', '_{}.log'.format(args.partition_id))), filemode='w', level=logging.DEBUG)

    subject_list =  path_utils.get_subject_list()
    subject_list = np.sort(subject_list)[args.partition_id::args.num_partitions]


    FORCE_OVERRIDE = False
    fail_list = []
    noise_classifier_obj = noise_classifier()
    for subject in subject_list:#tqdm(subject_list):
        logging.info('\nworking on {}\n'.format(subject))
        paths = path_utils.get_paths(subject=subject, mode='bipolar')
        for path in paths:
            annot_fname = path_utils.target_file_name(path['signals'], 'annot')
            if FORCE_OVERRIDE or (not os.path.isfile(annot_fname)):
                try:
                    # read .edf file and generate mne_object with annotation of bad chans and segs
                    mne_wrapper = my_mne_wrapper()
                    mne_wrapper.read_edf_file(path['signals'])  # ), chanel_groups=electrodes)
                    mne_wrapper.preprocess()
                    assert mne_wrapper.get_mne().info['sfreq'] == 500
                    mne_copy = mne_wrapper.get_mne().copy()

                    # read the annotations
                    event_obj = event_reader(fname=path['events'])
                    event_obj.align_to_sampling_rate(old_sfreq=mne_wrapper.original_sfreq, new_sfreq=mne_copy.info['sfreq'])
                    read_success = True


                    add_annotations(mne_copy, event_obj.get_countdowns(), 'CNTDWN')
                    add_annotations(mne_copy, event_obj.get_list_events(), 'LIST')
                    add_annotations(mne_copy, event_obj.get_recalls(), 'RECALL')
                    add_annotations(mne_copy, event_obj.get_distracts(), 'DSTRCT')
                    add_annotations(mne_copy, event_obj.get_rests(), 'REST')
                    add_annotations(mne_copy, event_obj.get_orients(), 'ORIENT')
                    add_annotations(mne_copy, event_obj.get_word_events(), 'WORD')

                    # identify and mark bad chans and segs
                    mne_copy = noise_classifier_obj.mark_bad_chans_and_segs(mne_copy, logger=logging.getLogger(__name__))

                    annotate_obj = dict({'bads': mne_copy.info['bads'], 'annotations': mne_copy.annotations,
                                        'events': {'cntdwn': event_obj.get_countdowns(), 'list': event_obj.get_list_events(),
                                                    'recall': event_obj.get_recalls(), 'dstrct': event_obj.get_distracts(),
                                                    'rest': event_obj.get_rests(), 'orient': event_obj.get_orients(),
                                                    'words': event_obj.get_word_events()}})



                    os.makedirs(os.path.dirname(annot_fname), exist_ok=True)
                    with open(annot_fname, 'wb') as fd:
                        pickle.dump(annotate_obj, fd)
                        #json.dump(annotate_obj, fd)

                except:
                    fail_list.append(path['signals'])
                    logging.warning('FAILED TO GENERATE .annot  FROM   {}'.format(path['signals']))


    if len(fail_list) > 0:
        print('FAILED TO GENERATE THE FOLLOWING FILES:')
        for fname in fail_list:
            print('\t', fname)

    #print('total channels:   {}   excluded channels:  {}'.format(total_chans, exclude_chans))