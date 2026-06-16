import copy
import glob
import os

import mne
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import tqdm

from path_utils import get_subject_list, get_paths
from event_reader import event_reader
from my_montage_reader import my_montage_reader
from my_mne_wrapper import my_mne_wrapper
from paths_and_constants import *


def make_data_availability_list(region_list, hemisphere_sel):

    subject_list = get_subject_list()
    data = dict()

    for i_subject, subject in enumerate(subject_list):

        paths = get_paths(subject=subject, sess_slct=None, mode='bipolar')
        # PATCH
        if len(paths) == 0:
            continue

        try:
            #
            subject_data = dict()

            # add contacts info
            montage = my_montage_reader(fname=paths[0]['electrodes'])
            electrode_list = montage.get_electrode_list_by_region(region_list=region_list, hemisphere_sel=hemisphere_sel)
            subject_data['electrodes'] = electrode_list
            total_contacts = np.sum([len(electrode_list[region]) for region in electrode_list])

            # add session info
            subject_data['numsessions'] = len(paths)
            subject_data['sessions'] = dict()
            for i_sess, path in enumerate(paths):
                sess_name = path['events'][path['events'].find('ses-'):][:5]
                #
                event_reader_obj = event_reader(path['events'])
                num_countdowns = len(event_reader_obj.get_countdowns())
                #
                mne_data = mne.io.read_raw_edf(path['signals'], preload=False)
                subject_data['bipolar_names'] = mne_data.ch_names
                #
                subject_data['sessions'][sess_name] = {'date': mne_data.info['meas_date'], 'num countdowns': num_countdowns}

            if total_contacts > 0:
                data[subject] = subject_data
                print(subject, '  ({} / {})'.format(i_subject + 1, len(subject_list)))
            else:
                print('no contacts in ROI for', subject)

        except:
            print('failed for', subject)

    return data
    # generate excel file
    df = pd.DataFrame.from_dict(data).T
    # "flatening" the contact counts - failed to do it so I made it flat in the first place
    writer = pd.ExcelWriter('E:/ds004789-download/data_availability_for_protocol_1.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='.'.join([r[::2][:4] for r in region_list])[:31])
    writer.close()

def process_raw_data(data, ovrd_times=None):

    subjects_list = list(data.keys())

    miss_cnt, hit_cnt, deltas_list = 0, 0, []
    for subject in subjects_list:
        subject_data = data[subject]

        # map bipolar contacts to regions
        # lets create dictionary by electrode name
        electrodes = dict()
        bipolar_contacts = dict()
        for region in list(subject_data['electrodes'].keys()):
            for electrode in subject_data['electrodes'][region]:
                electrodes[electrode['name']] = {'region': region, 'coords': electrode['coords']}
            bipolar_contacts[region] = [] # we will fill that in next loop
        subject_data['electrodes by name'] = copy.deepcopy(electrodes)
        # maps
        bipolar_contacts['mixed'], bipolar_contacts['unresoved'] = [], []
        for name in subject_data['bipolar_names']:
            name1, name2 = name.split('-')
            try:
                el1, el2 = electrodes[name1], electrodes[name2]
            except:
                bipolar_contacts['unresoved'].append(name)
            not_boundary = el1['region'] == el2['region']
            if not_boundary:
                bipolar_contacts[el1['region']].append({'name1': name1, 'name2': name2, 'coords1': el1['coords'], 'coords2': el2['coords']})
            else:
                bipolar_contacts['mixed'].append([el1, el2])
        subject_data['bipolar contacts'] = copy.deepcopy(bipolar_contacts)


        # add time between sessions
        sess_names = list(subject_data['sessions'].keys())
        #print('')
        for i_sess, name in enumerate(sess_names):
            if ovrd_times is not None:
                locate = (ovrd_times['subject'].values == subject) * (ovrd_times['session'].values == int(name[-1])) * [ovrd_times['experiment'].values == 'FR1']
                i = np.argwhere(locate.squeeze()).squeeze()
                # subject_data['sessions'][name]['date'] = ovrd_times.iloc[i]['datetime']
                if i.size != 1:
                    miss_cnt += 1
                    print(subject, name, i.size)
                    for i1 in np.argwhere((ovrd_times['subject'].values == subject) * (ovrd_times['session'].values == int(name[-1]))).flatten():
                        print('\t', i1, ovrd_times.iloc[i1]['experiment'])
                else:
                    hit_cnt += 1
                    subject_data['sessions'][name]['date'] = datetime.datetime.strptime(ovrd_times.iloc[i]['datetime'], '%Y-%m-%d %H:%M:%S')
                    print(subject, name, i.size, '--- OK', subject_data['sessions'][name]['date'])
                #subject_data['sessions'][name]['date'] = datetime.datetime.fromtimestamp(ovrd_times.iloc[i]['mstime'] / 1000)# TBD fix timezone
            if i_sess == 0:
                base_date_str = subject_data['sessions'][sess_names[0]]['date']
                base_timestamp = datetime.datetime.timestamp(base_date_str)
            subject_data['sessions'][name]['timestamp'] = datetime.datetime.timestamp(subject_data['sessions'][name]['date'])
            subject_data['sessions'][name]['relative timestamp'] = subject_data['sessions'][name]['timestamp'] - base_timestamp
            #print(subject, name, subject_data['sessions'][name]['date'])
            if subject_data['sessions'][name]['relative timestamp'] != 0:
                print(subject_data['sessions'][name]['relative timestamp'] / 3600)
                deltas_list.append(subject_data['sessions'][name]['relative timestamp'])

    print('miss: {}   hit:  {}'.format(miss_cnt, hit_cnt))
    deltas_list = np.array(deltas_list) / 3600
    deltas_list = deltas_list[(np.abs(deltas_list) < 1000) * (np.abs(deltas_list) >= 1)]
    h = np.histogram(np.array(deltas_list), bins=np.linspace(start=-100, stop=1000, num=1101))
    plt.bar(h[1][1:], h[0])
    plt.grid(True)
    plt.xlabel('hours from sess-0')
    plt.show()

    return data


def make_availity_list_by_rules(data, region_list, min_sessions=3, countdown_range=[26, 26], min_total_bipolar=1):

    output = dict()
    for subject in list(data.keys()):
        print(subject)
        subject_data = data[subject]
        session_list, bipolar_contact_list = dict(), dict()
        num_bipolar_contacts = 0
        for region in region_list:
            n = len(subject_data['bipolar contacts'][region])
            if n >= 1:
                num_bipolar_contacts += n
                bipolar_contact_list[region] = subject_data['bipolar contacts'][region]

        for session_name in subject_data['sessions']:
            session = subject_data['sessions'][session_name]
            n = session['num countdowns']
            if (n >= countdown_range[0]) and (n <= countdown_range[-1]):
                session_list[session_name] = session

        if (num_bipolar_contacts >= min_total_bipolar) and (len(session_list) >= min_sessions):
            output[subject] = {'bipolar contacts': bipolar_contact_list, 'sessions': session_list}

    return output


class data_availability:

    base_folder = IDXS_FOLDER
    processed_folder = PROC_FOLDER

    def __init__(self, base_folder=IDXS_FOLDER, fixed_data_filename='fixed_availability_data'):

        with open(os.path.join(base_folder, fixed_data_filename), 'rb') as fd:
            self.data = pickle.load(fd)

    def __bipolar_contact_additional_data(self, contact_name, contact_details):

        names = contact_name.split('-')
        ext_data = []
        for name in names:
            try:
                ext_data.append(contact_details[name])
                ext_data[-1]['name'] = name
            except:
                ext_data.append({'region': '', 'coords': None, 'name': name})

        return ext_data




    def get_contacts_for_2_session_gap(self, min_timegap_hrs, max_timegap_hrs, event_type=None, sub_event_type=None, epoch_subset=None, enforce_first=False, single_session=False):

        # first make flattened session list
        suitable_session_pairs = []
        for subject in self.data:
            subject_data = self.data[subject]
            keys = list(subject_data['sessions'].keys())
            if single_session:
                # PATCH TO USE THE CODE FOR SINGLE SESSION
                keys, min_timegap_hrs = [keys[0], keys[0]], -1
            num_sessions = len(keys)
            timegap_matrix_hrs = np.zeros((num_sessions, num_sessions))
            session_dates =  [subject_data['sessions'][sess]['date'] for sess in keys]
            for i1 in range(num_sessions-1):
                for i2 in range(i1+1, num_sessions):
                    try:
                        timegap_matrix_hrs[i1, i2] = (session_dates[i2] - session_dates[i1]).total_seconds() / 3600
                    except:
                        print('fail', subject, keys[i1], keys[i2], session_dates[i2], session_dates[i1])
            #
            suitability_mat = (timegap_matrix_hrs > min_timegap_hrs) * (timegap_matrix_hrs < max_timegap_hrs)
            for i1 in range(1 if enforce_first else num_sessions-1):
                for i2 in range(i1+1, num_sessions):
                    if suitability_mat[i1, i2]:
                        pair = {'subject': subject, 'first': keys[i1], 'second': keys[i2],
                                'delta_hrs': timegap_matrix_hrs[i1, i2], 'contacts': subject_data['bipolar_names'], 'contacts_details': subject_data['electrodes by name']}
                        suitable_session_pairs.append(pair)
                        suitability_mat[i1, :], suitability_mat[:, i1], suitability_mat[i2, :], suitability_mat[:, i2] = False, False, False, False

        # intersect with available evoked files
        if event_type is not None:

            #pattern = os.path.join(self.processed_folder, 'sub-R*', 'ses-*', event_type, '*' + sub_event_type + '-ieeg-evoked-ave.fif')
            if epoch_subset is None:
                pattern = os.path.join(self.processed_folder, 'sub-R*', 'ses-*', event_type, '*bipolar_-' + sub_event_type + '-ieeg-evoked-ave.fif')
            else:
                pattern = os.path.join(self.processed_folder, 'sub-R*', 'ses-*', event_type, '*bipolar_{}--'.format(epoch_subset) + sub_event_type + '-ieeg-evoked-ave.fif')
            evoked_list = glob.glob(pattern)
            revised_pair_list = []
            for pair in suitable_session_pairs:
                subject, first, second = pair['subject'], pair['first'], pair['second']
                # look for the files
                evoked_1, evoked_2 = None, None
                for fname in evoked_list:
                    if (fname.find(subject) > -1):
                        evoked_1 = fname if fname.find(first) > -1 else evoked_1
                        evoked_2 = fname if fname.find(second) > -1 else evoked_2
                if (evoked_1 is not None) and (evoked_2 is not None):
                    pair['first'] = evoked_1
                    pair['second'] = evoked_2
                    revised_pair_list.append(pair)
            #print([p['subject'] for p in revised_pair_list])
            suitable_session_pairs = revised_pair_list


        # flatten
        suitable_contacts = []
        for pair in suitable_session_pairs:
            for contact in pair['contacts']:
                contact_data = {'subject': pair['subject'], 'name': contact, 'first': pair['first'], 'second': pair['second'], 'delta_hrs': pair['delta_hrs']}
                contact_data['location'] = self.__bipolar_contact_additional_data(contact, pair['contacts_details'])
                suitable_contacts.append(contact_data)

        return suitable_session_pairs, suitable_contacts






    def get_get_contacts_for_2_session_gap_epoch_splits(self, min_timegap_hrs, max_timegap_hrs, event_type=None, sub_event_type=None, epoch_subsets=None, enforce_first=False, single_session=False):

        _, contact_list = self.get_contacts_for_2_session_gap(min_timegap_hrs=min_timegap_hrs, max_timegap_hrs=max_timegap_hrs,
                                                              event_type=event_type, sub_event_type=sub_event_type, epoch_subset=epoch_subsets[0], single_session=single_session)

        for contact in contact_list:
            contact['first'], contact['second'] = [contact['first']], [contact['second']]

        for i_sbst, second_epoch_subset in enumerate(epoch_subsets[1:]):

            _, contact_list2 = self.get_contacts_for_2_session_gap(min_timegap_hrs=min_timegap_hrs, max_timegap_hrs=max_timegap_hrs,
                                                                   event_type=event_type, sub_event_type=sub_event_type, epoch_subset=second_epoch_subset,
                                                                   enforce_first=enforce_first, single_session=single_session)
            # TBD find intersect of two lists
            combined_list = []
            rplc_pattrn_1, rplc_pattern2 = str(epoch_subsets[0]), str(second_epoch_subset)
            for i_cntct, contact in tqdm.tqdm(enumerate(contact_list)):
                # check if the same contact apears in the second list
                exist2 = False
                for i_cntct2, contact2 in enumerate(contact_list2):
                    if contact['subject'] == contact2['subject']:
                        if contact['name'] == contact2['name']:
                            ok1 = contact['first'][0].replace(rplc_pattrn_1, rplc_pattern2) == contact2['first']
                            ok2 = contact['second'][0].replace(rplc_pattrn_1, rplc_pattern2) == contact2['second']
                            if ok1 and ok2:
                                contact['first'].append(contact2['first'])
                                contact['second'].append(contact2['second'])
                                combined_list.append(contact)
                                break
            contact_list = combined_list

        return contact_list


    def get_subject_list(self):

        return list(self.data.keys())


    def get_sessions_for_subject(self, subject, event_type='CNTDWN', sub_event_type='', epoch_subsets=None, at_leatst_2_session=True):

        subject_data = self.data[subject]
        if at_leatst_2_session and (len(subject_data['sessions']) < 2):
            return []

        elapsed = np.diff([subject_data['sessions'][s]['timestamp'] for s in subject_data['sessions']]) / 3600
        elapsed = np.concatenate(([0], elapsed))

        assert epoch_subsets is not None
        evoked_lists = []
        for epoch_subset in epoch_subsets:
            pattern = os.path.join(self.processed_folder, subject, 'ses-*', event_type.lower(), '*bipolar_{}--{}'.format(epoch_subset, event_type + '-ieeg-evoked-ave.fif'))
            evoked_lists.append(glob.glob(pattern))

        # availability matrix
        sessions = list(subject_data['sessions'])
        availability_matrix = np.zeros((len(epoch_subsets), subject_data['numsessions']), dtype=bool)
        for i_epoch, epoch_subset in enumerate(epoch_subsets):
            for i_sess, sess in enumerate(sessions):
                availability_matrix[i_epoch, i_sess] = np.any([name.find(sess) > -1 for name in evoked_lists[i_epoch]])

        # find best session combination
        session_sel_mask = availability_matrix.sum(axis=0) >= len(epoch_subsets)
        if (session_sel_mask.sum() < 2) or (not session_sel_mask[0]):
            return []

        # make lists for all SESSION-PAIRS (to be able to use pair code)
        pair_lists = []
        i1 = np.argwhere(session_sel_mask).flatten()[0]
        for i2 in i1 + 1 + np.argwhere(session_sel_mask[i1 + 1:]).flatten():
            first_idxs = [np.argwhere([e.find(sessions[i1] + '_') > -1 for e in evoked_lists[i]]).squeeze() for i in range(len(epoch_subsets))]
            second_idxs = [np.argwhere([e.find(sessions[i2] + '_') > -1 for e in evoked_lists[i]]).squeeze() for i in range(len(epoch_subsets))]
            #print(i1, i2, subject)
            suitable_contacts = []
            for i_cntct, contact in enumerate(subject_data['bipolar_names']):
                suitable_contacts.append({'subject': subject, 'name': contact, 'delta_hrs': elapsed[i2],
                                          'location': self.__bipolar_contact_additional_data(contact, subject_data['electrodes by name']),
                                          'first': [evoked_lists[i][first_idxs[i]] for i in range(len(epoch_subsets))],
                                          'second': [evoked_lists[i][second_idxs[i]] for i in range(len(epoch_subsets))]})
            pair_lists.append(suitable_contacts)

        #
        return pair_lists



    # def get_avail_evoked_ave_files(self, major_event, sub_event):
    #
    #     pattern = os.path.join(self.processed_folder, 'sub-R*', 'ses-*', major_event, '*' + sub_event + '-ieeg-evoked-ave.fif')
    #
    #     return glob.glob(pattern)


class contact_list_services:

    def __init__(self):

        pass

    def is_same_contact(self, c1, c2):
        same_sbjct = c1['subject'] == c2['subject']
        same_name = c1['name'] == c2['name']
        return same_sbjct and same_name

    def intersect_lists(self, list1, list2):

        list = []
        for c1 in list1:
            # for c2 in list2:
            #     if same_contact(c1, c2):
            #         list.append(c1)
            if np.any([self.is_same_contact(c1, c2) for c2 in list2]):
                list.append(c1)

        return self.remove_double_contacts(list)

    def remove_double_contacts(self, list):

        new_list = []
        for c1 in list:
            # exists = False
            # for c2 in new_list:
            #     exists = exists or same_contact(c1, c2)
            # if not exists:
            #     new_list.append(c1)
            if not np.any([self.is_same_contact(c1, c2) for c2 in new_list]):
                new_list.append(c1)

        return new_list


    def get_sesseion_list(self, contact_list):

        sess_list = dict()
        for c in contact_list:
            subject = c['subject']
            sess0 = c['first'][0][c['first'][0].find('ses-'):][:5]
            sess1 = c['second'][0][c['second'][0].find('ses-'):][:5]
            #print(subject, sess0, sess1)
            if not (subject in list(sess_list.keys())):
                sess_list[subject] = dict({sess0: 0, sess1: 0})
            sess_list[subject][sess0] += 1
            sess_list[subject][sess1] += 1

        return sess_list



if __name__ == '__main__':

    # data_availability_obj = data_availability()
    # _, suitable_contacts = data_availability_obj.get_contacts_for_2_session_gap(min_timegap_hrs=72, max_timegap_hrs=96)
    # print(len(suitable_contacts))

    READ_FILES = True

    base_folder = 'E:/ds004789-download'

    start_times = pd.read_csv(os.path.join(base_folder, 'start_times.csv'))

    # area of interest definition
    all_regions = ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'frontalpole', 'fusiform',
                   'inferiorparietal', 'inferiortemporal', 'insula', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal',
                   'lingual', 'medialorbitofrontal', 'middletemporal', 'nan', 'paracentral', 'parahippocampal', 'parsopercularis',
                   'parsorbitalis', 'parstriangularis', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
                   'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal',
                   'supramarginal', 'temporalpole', 'transversetemporal']
    all_regions_hemisphere_sel = ['LR' for region in all_regions]
    # create table of available data
    if READ_FILES:
        subjects_list = get_subject_list()
        raw_availability_data = make_data_availability_list(region_list=all_regions, hemisphere_sel=all_regions_hemisphere_sel)
        with open(os.path.join(base_folder, 'raw_availability_data'), 'wb') as fd:
            pickle.dump(raw_availability_data, fd)
        #assert False

    # process the files
    with open(os.path.join(base_folder, 'raw_availability_data'), 'rb') as fd:
        raw_availability_data = pickle.load(fd)
    availability_data = process_raw_data(raw_availability_data, ovrd_times=start_times)

    #
    with open(os.path.join(base_folder, 'fixed_availability_data'), 'wb') as fd:
        pickle.dump(availability_data, fd)
    #

    region_list = ['fusiform-R', 'inferiortemporal-R', 'lateraloccipital-R', 'lingual-R', 'fusiform-L', 'inferiortemporal-L', 'lateraloccipital-L', 'lingual-L']
    #hemisphere_sel = ['LR', 'LR', 'LR', 'both']
    admit_list = make_availity_list_by_rules(availability_data, region_list, min_sessions=3, countdown_range=[26, 26], )
    print(admit_list.keys())


