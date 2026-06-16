import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class event_reader:

    def __init__(self, fname):

        assert fname[-4:] == '.tsv'
        self.df = pd.read_csv(fname, delimiter='\t')

        # onsets = event_df['onset']
        # durations = self.df['duration']
        # satrt_samples = self.df['sample']
        # types = self.df['trial_type']
        # neames = self.df['onset']
        # word_ids = self.df['serialpos']
        # list_ids = self.df['list']
        # tests = self.df['test']  # WHAT THAT IS?
        # responce_times = self.df['response_time']  # WHAT THAT IS?
        # answers = self.df['answer']  # WHAT THAT IS?


    def align_to_sampling_rate(self, old_sfreq=0, new_sfreq=0):

        # save_option = pd.options.mode.copy_on_write
        # pd.options.mode.copy_on_write = True
        for i_line in range(self.df.shape[0]):
            self.df.at[i_line, 'sample'] = int(self.df.iloc[i_line]['sample'] * (new_sfreq / old_sfreq))
        # pd.options.mode.copy_on_write = save_option


    def get_countdowns(self):

        countdowns = []

        countdown_start_idxs = np.argwhere(self.df['trial_type'] == 'COUNTDOWN_START').squeeze()
        countdown_end_idxs = np.argwhere(self.df['trial_type'] == 'COUNTDOWN_END').squeeze()
        for i_cntdn, (start_idx, stop_idx) in enumerate(zip(countdown_start_idxs, countdown_end_idxs)):
            event = dict()
            line = self.df.iloc[start_idx]
            event['onset'] = line['onset']
            event['onset sample'] = line['sample']
            event['interim events'] = None
            for i1 in range(start_idx + 1, stop_idx):
                line = self.df.iloc[i1]
                event_line = '{} \t time {:6.1}  sample {}'.format(line['trial_type'], line['onset'], line['sample'])
                event['interim events'] = [event_line] if event['interim events'] is None else event['interim events'] + [event_line]
            line = self.df.iloc[stop_idx]
            event['end'] = line['onset']
            event['end sample'] = line['sample']
            countdowns.append(event)

        return countdowns

    def get_recalls(self, random_start=False, max_duration=None):

        recalls = []
        rec_start_idxs = np.argwhere(self.df['trial_type'] == 'REC_START').flatten()
        rec_end_idxs = np.argwhere(self.df['trial_type'] == 'REC_END').flatten()
        rec_word_idxs = np.argwhere((self.df['trial_type'] == 'REC_WORD') + (self.df['trial_type'] == 'REC_WORD_VV')).flatten()
        for i_rec, (start_idx, stop_idx) in enumerate(zip(rec_start_idxs, rec_end_idxs)):
            event = dict()
            line = self.df.iloc[start_idx]
            event['onset'] = line['onset']
            event['onset sample'] = line['sample']
            event['interim events'] = None
            for i1 in range(start_idx + 1, stop_idx):
                line = self.df.iloc[i1]
                event_line = '{} \t time {:6.1}  sample {}'.format(line['trial_type'], line['onset'], line['sample'])
                event['interim events'] = [event_line] if event['interim events'] is None else event['interim events'] + [event_line]
            line = self.df.iloc[stop_idx]
            event['end'] = line['onset']
            event['end sample'] = line['sample']
            if max_duration is not None:
                event['original boundaries'] = {'onset': event['onset'], 'onset sample': event['onset sample'], 'end': event['end'], 'end sample': event['end sample']}
                margin = event['end'] - event['onset'] - max_duration
                if random_start:
                    if margin > 0:
                        dstart = np.random.uniform(low=0, high=margin)
                    else:
                        dstart = 0
                dend = margin - dstart if margin > 0 else 0
                event['onset'] += dstart
                event['end'] -= dend
                event['onset sample'] += int(dstart * 500)
                event['end sample'] -= int(dend * 500)
            else:
                assert not random_start # random start must have max_duration, which actually becomes target duration
            recalls.append(event)

        return recalls

    def get_distracts(self, random_start=False, max_duration=None):

        dstrcts = []
        dstrct_start_idxs = np.argwhere(self.df['trial_type'] == 'DISTRACT_START').flatten()
        dstrct_end_idxs = np.argwhere(self.df['trial_type'] == 'DISTRACT_END').flatten()
        for i_dstrct, (start_idx, stop_idx) in enumerate(zip(dstrct_start_idxs, dstrct_end_idxs)):
            event = dict()
            line = self.df.iloc[start_idx]
            event['onset'] = line['onset']
            event['onset sample'] = line['sample']
            event['interim events'] = None
            for i1 in range(start_idx + 1, stop_idx):
                line = self.df.iloc[i1]
                event_line = '{} \t time {:6.1}  sample {}'.format(line['trial_type'], line['onset'], line['sample'])
                event['interim events'] = [event_line] if event['interim events'] is None else event['interim events'] + [event_line]
            line = self.df.iloc[stop_idx]
            event['end'] = line['onset']
            event['end sample'] = line['sample']
            if max_duration is not None:
                event['original boundaries'] = {'onset': event['onset'], 'onset sample': event['onset sample'], 'end': event['end'], 'end sample': event['end sample']}
                margin = event['end'] - event['onset'] - max_duration
                if random_start:
                    if margin > 0:
                        dstart = np.random.uniform(low=0, high=margin)
                    else:
                        dstart = 0
                dend = margin - dstart if margin > 0 else 0
                event['onset'] += dstart
                event['end'] -= dend
                event['onset sample'] += int(dstart * 500)
                event['end sample'] -= int(dend * 500)
            else:
                assert not random_start # random start must have max_duration, which actually becomes target duration
            dstrcts.append(event)

        return dstrcts

    def get_rests(self, random_start=False, max_duration=None):

        # rests usualy apear between orient and lists, between distract and recalls, and after rec
        rests = []
        orients = self.get_orients()
        lists = self.get_list_events()
        distracts = self.get_distracts()
        recalls = self.get_recalls()
        all_onsets = np.array(self.df['onset'])

        rec_start_idxs = np.argwhere(self.df['trial_type'] == 'REC_START').flatten()
        prelim_rest_onsets, prelim_rest_ends = [], []

        # between orients and lists
        orient_ends = np.array([e['end'] for e in orients])
        list_onsets = np.array([e['onset'] for e in lists])
        for orient_end in orient_ends:
            next_list_onset = list_onsets[list_onsets > orient_end]
            if next_list_onset.size > 0:
                next_list_onset = next_list_onset[0]
                gap = next_list_onset - orient_end
                if gap < 10:
                    prelim_rest_onsets.append(orient_end)
                    prelim_rest_ends.append(next_list_onset)

        # between distract and recalls
        distract_ends = np.array([e['end'] for e in distracts])
        recall_onsets = np.array([e['onset'] for e in recalls])
        for distract_end in distract_ends:
            next_recall_onset = recall_onsets[recall_onsets > distract_end]
            if next_recall_onset.size > 0:
                next_recall_onset = next_recall_onset[0]
                gap = next_recall_onset - distract_end
                if gap < 25:
                    prelim_rest_onsets.append(distract_end)
                    prelim_rest_ends.append(next_recall_onset)

        # after recs
        recall_ends = np.array([e['end'] for e in recalls])
        for recall_end in recall_ends:
            idx = np.argwhere(recall_end == all_onsets).flatten()[-1]
            if idx < self.df.shape[0] - 1:
                next_event = self.df.iloc[idx + 1]
                #print(recall_end, next_event['trial_type'], next_event['onset'], next_event['onset'] - recall_end)
                prelim_rest_onsets.append(recall_end)
                prelim_rest_ends.append(next_event['onset'])

        onset_idxs, end_idxs = [], []
        for t_onset, t_end in zip(prelim_rest_onsets, prelim_rest_ends):
            idx_onset = np.argwhere(t_onset == all_onsets).squeeze()
            idx_end = np.argwhere(t_end == all_onsets).squeeze()
            try:
                idx_end = idx_end[0] if idx_end.size > 1 else idx_end
                if idx_end - idx_onset > 1:
                    gap = t_end - t_onset
                    mid = t_onset + gap / 2
                    middle = all_onsets[idx_onset + 1:idx_end]
                    from_onset = middle - t_onset
                    to_end = t_end = middle
                    # TBD deal with this
                else:
                    onset_idxs.append(idx_onset)
                    end_idxs.append(idx_end)
            except:
                pass

        lengths = np.array([self.df.iloc[end_idx]['onset'] - self.df.iloc[onset_idx]['onset'] for onset_idx, end_idx in zip(onset_idxs, end_idxs)])
        min_len, max_len = 5, 20
        mask = (lengths > min_len) * (lengths < max_len)

        for onset_idx, end_idx in zip(np.array(onset_idxs)[mask], np.array(end_idxs)[mask]):
            event = {'onset': self.df.iloc[onset_idx]['onset'], 'onset sample': self.df.iloc[onset_idx]['sample'], 'interim events': None,
                     'end': self.df.iloc[end_idx]['onset'], 'end sample': self.df.iloc[end_idx]['sample']}
            rests.append(event)


        return rests


    def get_orients(self):

        orients = []

        guess_orient_duration = 8
        guess_fs = (self.df['sample'][self.df.shape[0] - 1] - self.df['sample'][0]) / (self.df.onset[self.df.shape[0] - 1] - self.df.onset[0])

        orient_start_idxs = np.argwhere(self.df['trial_type'] == 'ORIENT_START').squeeze()
        orient_end_idxs = np.argwhere(self.df['trial_type'] == 'ORIENT_END').squeeze()
        orient_idxs = np.argwhere(self.df['trial_type'] == 'ORIENT').squeeze()
        all_orient_starts = np.concatenate((orient_start_idxs, orient_idxs))
        for i_orient, start_idx in enumerate(all_orient_starts):
            event = dict()
            line = self.df.iloc[start_idx]
            event['onset'] = line['onset']
            event['onset sample'] = line['sample']
            event['interim events'] = None
            # try to figure out if there is an end event
            orient_end_idx = None
            possible_ends = orient_end_idxs[orient_end_idxs > line['onset']]
            if possible_ends.size > 0:
                orient_end_idx = possible_ends[0]
                if i_orient < all_orient_starts.size - 1:
                    orient_end_idx = orient_end_idx if orient_end_idx < all_orient_starts[i_orient + 1] else None
            if orient_end_idx is None:
                event['end'] = line['onset'] + guess_orient_duration
                event['end sample'] = int(event['onset sample'] + guess_orient_duration * guess_fs)
            else:
                line = self.df.iloc[orient_end_idx]
                event['end'] = line['onset']
                event['end sample'] = line['sample']

            orients.append(event)

        return orients


    def get_word_events(self):

        #prectice_event_ids = np.argwhere(self.df['trial_type'] == 'PRACTICE_WORD').flatten()

        def read_word_event(line):

            return({'word': line['item_name'], 'list': line['list'], 'serial': line['serialpos'], 'onset': line['onset'], 'onset sample': line['sample'], 'duration': line['duration']})

        word_event_ids = np.argwhere(self.df['trial_type'] == 'WORD').flatten()
        word_recall_ids =  np.argwhere(self.df['trial_type'] == 'REC_WORD').flatten()

        lists = dict()
        # read all presented words
        for idx in word_event_ids:
            event = read_word_event(self.df.iloc[idx])
            list_id = event['list']
            if not list_id in list(lists.keys()):
                lists[list_id] = {'words': [], 'recall_words': [], 'events': []}
            lists[list_id]['words'].append(event['word'])
            lists[list_id]['events'].append(event)

        # read all recalled words
        for idx in word_recall_ids:
            event = read_word_event(self.df.iloc[idx])
            list_id = event['list']
            assert list_id in list(lists.keys())
            lists[list_id]['recall_words'].append(event['word'])

        # go over lists of presented words and mark each as recalled/not recaled
        for list_id in lists:
            lists[list_id]['recall_mask'] = np.zeros(len(lists[list_id]['words']), dtype=bool)
            for i, word in enumerate(lists[list_id]['words']):
                is_recall = word in lists[list_id]['recall_words']
                lists[list_id]['recall_mask'][i] = is_recall
                lists[list_id]['events'][i]['recall'] = is_recall


        # return lists

        # making a "flat" list (DO I NEED THIS???????)
        event_list = []
        for list_id in lists:
            for event in lists[list_id]['events']:
                event['end sample'] = event['onset sample'] + int(500 * event['duration'])
                event_list.append(event)

        return event_list

    def get_list_events(self):

        words = self.get_word_events()
        list_dict = dict()
        for w in words:
            list = w['list']
            if not (list in list_dict):
                list_dict[list] = copy.deepcopy(w)
                del list_dict[list]['recall']
                del list_dict[list]['serial']
                del list_dict[list]['word']
                list_dict[list]['end sample'] = list_dict[list]['onset sample'] + list_dict[list]['duration'] * 500
            else:
                new_onset = w['onset']
                if new_onset < list_dict[list]['onset']:
                    list_dict[list]['duration'] += (list_dict[list]['onset'] - new_onset)
                    list_dict[list]['onset'] = new_onset
                    list_dict[list]['onset sample'] = w['onset sample']
                else:
                    new_duration = new_onset + w['duration'] - list_dict[list]['onset']
                    list_dict[list]['duration'] = max(list_dict[list]['duration'], new_duration)
                    list_dict[list]['end sample'] = w['end sample']#list_dict[list]['onset sample'] + list_dict[list]['duration'] * 500

        return [list_dict[key] for key in list_dict]



    def make_timelines(self):

        max_cycle_duration = 120
        timeline_res = 0.2
        timeline_size = np.ceil(max_cycle_duration / timeline_res).astype(int)
        types = self.df['trial_type'].values
        onsets = self.df['onset'].values
        # identifying the cycles
        cycle_start_idxs = np.argwhere([t == 'COUNTDOWN_START' for t in types]).flatten()
        cycle_end_idxs = np.concatenate((cycle_start_idxs[1:], np.atleast_1d(len(types))))
        cycle_starts = onsets[[t == 'COUNTDOWN_START' for t in types]]
        cycle_boundaries = cycle_starts + onsets[-1]
        cycle_durations = np.diff(cycle_boundaries)
        sel_cycles = cycle_durations < max_cycle_duration

        indicator_matrix = np.zeros((sel_cycles.sum(), timeline_size), dtype=int)
        indicator_leftovers = []
        interim_list = []

        for i_out_cycle, i_in_cycle in enumerate(np.argwhere(sel_cycles).flatten()):
            cycle_start = cycle_starts[i_in_cycle]
            cycle_end = cycle_start + cycle_durations[i_in_cycle]
            cycle_start_idx = cycle_start_idxs[i_in_cycle]
            cycle_end_idx = cycle_end_idxs[i_in_cycle]
            sublist = types[cycle_start_idx:cycle_end_idx]
            time_in_cycle = onsets[cycle_start_idx:cycle_end_idx] - onsets[cycle_start_idx]
            sublist_indicator = [t == 'ORIENT' for t in sublist]
            ORIENT = time_in_cycle[np.argwhere(sublist_indicator).squeeze()] if np.any(sublist_indicator) else -1
            sublist_indicator = [t == 'TRIAL' for t in sublist]
            TRIAL = time_in_cycle[np.argwhere(sublist_indicator).squeeze()] if np.any(sublist_indicator) else -1
            sublist_indicator = [t == 'WORD' for t in sublist]
            sublist_idxs = np.argwhere(sublist_indicator).flatten()
            if sublist_idxs.size > 0:
                WORD_START_IDX = sublist_idxs[0]
                # print(i_in_cycle, i_out_cycle, cycle_start, cycle_end, cycle_start_idx, cycle_end_idx, sublist_idxs[0], sublist_idxs[0])
                WORD_START = time_in_cycle[sublist_idxs[0]]
                WORD_END = time_in_cycle[sublist_idxs[-1] + 1]
                COUNTDOWN_START = time_in_cycle[np.argwhere([t == 'COUNTDOWN_START' for t in sublist]).squeeze()]
                COUNTDOWN_END = time_in_cycle[np.argwhere([t == 'COUNTDOWN_END' for t in sublist]).squeeze()]
                COUNTDOWN_END_IDX = np.argwhere([t == 'COUNTDOWN_END' for t in sublist]).squeeze()
                DISTRACT_START = time_in_cycle[np.argwhere([t == 'DISTRACT_START' for t in sublist]).squeeze()]
                DISTRACT_END = time_in_cycle[np.argwhere([t == 'DISTRACT_END' for t in sublist]).squeeze()]
                REC_START = time_in_cycle[np.argwhere([t == 'REC_START' for t in sublist]).squeeze()]
                REC_END = time_in_cycle[np.argwhere([t == 'REC_END' for t in sublist]).squeeze()]
                clr_legend = dict({'ORIENT': 7, 'WORD': 2, 'DISTRACT': 3, 'CNTDWN': 1, 'REC': 4, 'OUT': 5, 'TRIAL': 8, 'ENCODING': 6, 'PRACTICE': 10, 'WAITING': 9, 'NOTHING': 0})
                if ORIENT > 0:
                    indicator_matrix[i_out_cycle, np.round(ORIENT / timeline_res).astype(int)] = clr_legend['ORIENT']
                word_range = range(int(WORD_START / timeline_res), int(WORD_END / timeline_res))
                indicator_matrix[i_out_cycle, word_range] = clr_legend['WORD']
                distract_range = range(int(DISTRACT_START / timeline_res), int(DISTRACT_END / timeline_res))
                indicator_matrix[i_out_cycle, distract_range] = clr_legend['DISTRACT']
                contdown_range = range(int(COUNTDOWN_START / timeline_res), int(COUNTDOWN_END / timeline_res))
                indicator_matrix[i_out_cycle, contdown_range] = clr_legend['CNTDWN']
                rec_range = range(int(REC_START / timeline_res), int(REC_END / timeline_res))
                indicator_matrix[i_out_cycle, rec_range] = clr_legend['REC']
                next_cycle_range = range(int(cycle_durations[i_in_cycle] / timeline_res), int(max_cycle_duration / timeline_res))
                indicator_matrix[i_out_cycle, next_cycle_range] = clr_legend['OUT']
                #
                if np.all(TRIAL > 0):
                    indicator_matrix[i_out_cycle, (TRIAL / timeline_res).astype(int)] = clr_legend['TRIAL']
                TRIAL_START = np.argwhere([t == 'TRIAL_START' for t in sublist])
                TRIAL_END = np.argwhere([t == 'TRIAL_END' for t in sublist])
                if min(TRIAL_START.size, TRIAL_END.size) > 0:
                    trial_range = range(int(time_in_cycle[TRIAL_START.squeeze()] / timeline_res), int(time_in_cycle[TRIAL_END.squeeze()] / timeline_res))
                    indicator_matrix[i_out_cycle, trial_range] = clr_legend['TRIAL']
                #
                ORIENT_START = np.argwhere([t == 'ORIENT_START' for t in sublist])
                ORIENT_START = np.argwhere([t == 'ORIENT' for t in sublist]) if ORIENT_START.size == 0 else ORIENT_START
                ORIENT_END = np.argwhere([t == 'ORIENT_END' for t in sublist])
                ORIENT_END = np.argwhere([t == 'ORIENT_OFF' for t in sublist]) if ORIENT_END.size == 0 else ORIENT_END
                if min(ORIENT_START.size, ORIENT_END.size) > 0:
                    orient_range = range(int(time_in_cycle[ORIENT_START.squeeze()] / timeline_res), int(time_in_cycle[ORIENT_END.squeeze()] / timeline_res))
                    indicator_matrix[i_out_cycle, orient_range] = clr_legend['ORIENT']
                #
                ENCODING_START = np.argwhere([t == 'ENCODING_START' for t in sublist])
                if ENCODING_START.size > 0:
                    indicator_matrix[i_out_cycle, int(time_in_cycle[ENCODING_START.squeeze()] / timeline_res)] = clr_legend['ENCODING']
                #
                PRACTICE_START = np.argwhere([t == 'PRACTICE_POST_INSTRUCT_START' for t in sublist])
                PRACTICE_END = np.argwhere([t == 'PRACTICE_POST_INSTRUCT_END' for t in sublist])
                if min(PRACTICE_START.size, PRACTICE_END.size) > 0:
                    practice_range = range(int(time_in_cycle[PRACTICE_START.squeeze()] / timeline_res), int(time_in_cycle[PRACTICE_END.squeeze()] / timeline_res))
                    indicator_matrix[i_out_cycle, practice_range] = clr_legend['PRACTICE']
                #
                WAITING_START = np.argwhere([t == 'WAITING_START' for t in sublist])
                WAITING_END = np.argwhere([t == 'WAITING_END' for t in sublist])
                if min(WAITING_START.size, WAITING_END.size) > 0:
                    WAITING_END = WAITING_END.flatten()[-1] if WAITING_END.size > 1 else WAITING_END
                    waiting_range = range(int(time_in_cycle[WAITING_START.squeeze()] / timeline_res), int(time_in_cycle[WAITING_END.squeeze()] / timeline_res))
                    indicator_matrix[i_out_cycle, waiting_range] = clr_legend['WAITING']
                #

                REC_END_IDX = np.argwhere([t == 'REC_END' for t in sublist]).squeeze()
                indicator_leftovers.append(list(sublist[REC_END_IDX+1:len(sublist)]))
                interim_list.append(list(sublist[COUNTDOWN_END_IDX+1:WORD_START_IDX]))

        x = np.arange(indicator_matrix.shape[1])
        xticks = (x[::50], x[::50] * timeline_res)

        return indicator_matrix, indicator_leftovers, interim_list, xticks, clr_legend




    def get_statistics(self):

        cnt_CNTDWN = (self.df['trial_type'] == 'COUNTDOWN_START').values.sum()
        cnt_WORD = (self.df['trial_type'] == 'WORD').values.sum()
        stat = {'total events': self.df.shape[0], 'last onset': self.df['onset'].values[-1], 'cntdwns': cnt_CNTDWN, 'words': cnt_WORD}

        return stat




        # word_dictionary = self.df['item_name'][prectice_event_ids].tolist()
        # word_dictionary = dict()
        # for i, line_id in enumerate(prectice_event_ids):
        #     line = self.df.iloc[line_id]
        #     word = line['item_name']
        #     event = {'onset': line['onset'], 'duration': line['duration'], 'onset sample': line['sample']}
        #     word_dictionary[word] = dict({'list': line['list'], 'serial': i, 'event': event, 'repeats': []})
        #
        # for line_id in word_event_ids:
        #     line = self.df.iloc[line_id]
        #     word = line['item_name']
        #     event = {'onset': line['onset'], 'duration': line['duration'], 'onset sample': line['sample']}
        #     if word in list(word_dictionary.keys()):
        #         word_dictionary[word]['lists'].append({'list': line['list'], 'serial': line['serial'], 'event': event})
        #     else:
        #         pass # TBD: wrong word repeats
        #
        # return word_dictionary


if __name__ == '__main__':

    base_folder = 'E:/ds004789-download'
    subject_list = ['sub-R1060M', 'sub-R1065J', 'sub-R1092J', 'sub-R1094T', 'sub-R1123C', 'sub-R1145J', 'sub-R1153T',
                    'sub-R1154D', 'sub-R1161E', 'sub-R1168T', 'sub-R1195E', 'sub-R1223E', 'sub-R1243T', 'sub-R1281E', 'sub-R1292E',
                    'sub-R1299T', 'sub-R1315T', 'sub-R1334T', 'sub-R1338T', 'sub-R1341T', 'sub-R1350D', 'sub-R1355T', 'sub-R1425D']

    from path_utils import get_paths

    images, leftovers, inters = [], [], []
    for subject in subject_list:
        print(subject)
        paths = get_paths(base_folder=base_folder, subject=subject)
        if len(paths) == 0:
            continue
        for path in paths:
            event_object = event_reader(path['events'])
            if len(event_object.get_countdowns()) == 26:
                img, lo, il, xticks, clr_legend = event_object.make_timelines()
                images.append(img)
                leftovers.append(lo)
                inters.append(il)

    #
    leftovers_flat = [l2 for l in leftovers for l1 in l for l2 in l1]
    print('types in between:', np.unique(leftovers_flat))
    interim_flat = [l2 for l in inters for l1 in l for l2 in l1]
    print('types between CNTDWN and WORD:', np.unique(interim_flat))
    #
    cmap = 'Paired'
    plt.imshow(np.concatenate(images), aspect='auto', cmap=cmap)
    plt.grid(True)
    plt.xticks(xticks[0], xticks[1])
    plt.show()
    #
    # re-align to end and make histograms
    REC_last, PRACTICE_last = [], []
    img = np.concatenate(images)
    for i in range(img.shape[0]):
        line = img[i]
        end_code = line[-1]
        b = np.argwhere(line == end_code).flatten()[0]
        line = np.concatenate((line[b:], line[:b]))
        img[i] = line
        #
        if True:#np.any(line == clr_legend['WAITING']):
            idxp = np.argwhere(line == clr_legend['PRACTICE'])
            idxr = np.argwhere(line == clr_legend['REC'])
            if idxp.size > 0:
                PRACTICE_last.append(idxp.flatten().max())
            elif idxr.size > 0:
                REC_last.append(idxr.flatten().max())
    plt.imshow(img, aspect='auto', cmap=cmap)
    plt.xticks(xticks[0], -np.flip(xticks[1]))
    plt.xlim([int(0.75 * img.shape[1]), img.shape[1]])
    plt.grid(True)
    plt.show()
    #
    h = np.histogram((img.shape[1] - np.array(REC_last)) * 0.2, bins=np.arange(start=0, stop=20, step=0.2))
    plt.bar(h[1][:-1], h[0], width=0.2)
    plt.grid(True)
    plt.show()
    #
    boundaries = np.arange(start=0, stop=img.shape[0], step=100)
    for istart, istop in zip(boundaries[:-1], boundaries[1:]):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img, aspect='auto', cmap=cmap)
        ax.set_xticks(xticks[0], -np.flip(xticks[1]))
        ax.set_xlim([int(0.75 * img.shape[1]), img.shape[1]])
        ax.set_ylim([istart, istop])
        ax.grid(True)
    plt.show()
    #
    codes, locs = np.unique(img.flatten(), return_index=True)
    codes = codes[np.argsort(locs)]
    annot = np.zeros(codes.size, dtype=np.object_)
    for key in clr_legend:
        idx = np.argwhere(clr_legend[key] == codes)
        if idx.size == 1:
            annot[idx] = key
    plt.imshow(np.atleast_2d(codes).T, aspect='auto', cmap=cmap)
    plt.yticks(np.arange(codes.size), annot)
    plt.show()
