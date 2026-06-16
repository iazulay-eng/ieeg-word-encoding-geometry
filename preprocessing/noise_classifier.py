import os
#import mne.preprocessing
import numpy as np
import matplotlib.pyplot as plt
import path_utils
#from my_montage_reader import my_montage_reader
import hurst
import seaborn as sns

from my_mne_wrapper import my_mne_wrapper
#from event_reader import event_reader

from paths_and_constants import *

# import logging
# logging.basicConfig(filename=os.path.join(BASE_FOLDER, os.path.basename(__file__).replace('.py', '.log')), filemode='w', level=logging.DEBUG)




class noise_classifier:

    def __init__(self):

        pass


    def __gaussian(self, bins, m, sigma, alpha):

        return alpha * np.exp(-(((bins - m) ** 2) / (2 * sigma ** 2)))


    def __fit_gaussian(self, bins, h, init_m, init_sigma):

        def calc_J(bins, h, m, sigma, alpha):

            g = self.__gaussian(bins, m, sigma, alpha)
            return np.sum((h - g) ** 2)

        m, sigma = init_m, init_sigma
        alpha = h.max()
        # return m, sigma, alpha
        for i in range(10):
            J0 = calc_J(bins, h, m, sigma, alpha)
            # print(i, J0, m, sigma, alpha)
            # optimise m
            srch_rng = 0.5 * sigma
            for i1 in range(10):
                srch_m = np.linspace(start=m - srch_rng, stop=m + srch_rng, num=11)
                J_vec = np.array([calc_J(bins, h, m0, sigma, alpha) for m0 in srch_m])
                m = srch_m[np.argmin(J_vec)]
                srch_rng *= 0.2
            # optimise sigma
            srch_rng = 0.2 * sigma
            for i1 in range(10):
                srch_sigma = np.linspace(start=sigma - srch_rng, stop=sigma + srch_rng, num=11)
                J_vec = np.array([calc_J(bins, h, m, s0, alpha) for s0 in srch_sigma])
                sigma = srch_sigma[np.argmin(J_vec)]
                srch_rng *= 0.2
            # optimise alpha
            srch_rng = 0.2 * alpha
            for i1 in range(10):
                srch_alpha = np.linspace(start=alpha - srch_rng, stop=alpha + srch_rng, num=11)
                J_vec = np.array([calc_J(bins, h, m, sigma, a0) for a0 in srch_alpha])
                alpha = srch_alpha[np.argmin(J_vec)]
                srch_rng *= 0.2

        return m, sigma, alpha


    def calc_hurst_lv_wrapper(self, mne_copy, num_seg=4, show=False, verbose=False, skip_hurst=False, ch_sel=None):

        # set skip hurst when debugging the complete flow, hurst consumes lots of time

        #fs = mne_copy.info['sfreq']
        mne_copy = mne_copy.copy().filter(l_freq=0.5, h_freq=100, fir_design='firwin', filter_length='auto', phase='zero', verbose=False)
        v = mne_copy.get_data()
        if ch_sel is not None:
            v = v[ch_sel]
        H_values = np.zeros((v.shape[0], num_seg))
        LV_values = np.zeros((v.shape[0], num_seg))
        steps = (np.linspace(0.01, 0.99, num=num_seg + 1) * v.shape[-1]).astype(int)
        for i_ch in range(v.shape[0]):
            for i_seg in range(num_seg):
                if skip_hurst:
                    H, c, data = 0.5, 1, None
                else:
                    H, c, data = hurst.compute_Hc(v[i_ch, steps[i_seg]:steps[i_seg+1]], kind='random_walk', simplified=False)
                H_values[i_ch, i_seg] = H
                LV_values[i_ch, i_seg] = np.log(v[i_ch, steps[i_seg]:steps[i_seg+1]].var())
                if verbose:
                    print('{} [{}]  :  {}  ,  {}'.format(i_ch, i_seg, H_values[i_ch, i_seg], LV_values[i_ch, i_seg]))

        H_scores = (H_values - H_values.mean()) / (H_values.std() + 1e-12)
        LV_scores = (LV_values - LV_values.mean()) / (LV_values.std() + 1e-12)

        if show:
            fig, ax = plt.subplots(1, 4)
            sns.heatmap(H_values, annot=True, ax=ax[0])
            sns.heatmap(LV_values, annot=True, ax=ax[2])
            sns.heatmap(np.abs(H_scores), annot=True, ax=ax[1])
            sns.heatmap(np.abs(LV_scores), annot=True, ax=ax[3])
            plt.show()

        return H_values, LV_values, H_scores, LV_scores


    def calc_lv_for_segments(self, mne_copy, t_seg=2, exclude_list=None):

        fs = mne_copy.info['sfreq']
        lseg = int(fs * t_seg)
        mne_copy = mne_copy.copy().filter(l_freq=0.5, h_freq=100, fir_design='firwin', filter_length='auto', phase='zero', verbose=False)
        v = mne_copy.get_data()
        if exclude_list is not None:
            chmask = np.ones(v.shape[0], dtype=bool)
            for i in exclude_list:
                chmask[i] = False
            v = v[chmask]
        nseg = np.ceil(v.shape[-1] / lseg).astype(int)
        boundaries = np.zeros(nseg+1, dtype=int)
        LV_values = np.zeros(nseg)
        for i_seg in range(nseg):
            istart, istop = i_seg*lseg, (i_seg+1)*lseg
            boundaries[i_seg], boundaries[i_seg+1] = istart, istop
            x = v[:, istart:istop]
            LV_values[i_seg] = np.log(x.var())
            #print(i_seg, '/', nseg)

        LV_score = (LV_values - LV_values.mean()) / LV_values.std()

        return LV_values, LV_score, boundaries




    def mark_bad_chans_and_segs(self, mne_copy, logger=None):

        mne_copy.filter(l_freq=0.1, h_freq=None, fir_design='firwin', filter_length='auto', phase='zero', verbose=False)  # REMOVE DC swing
        ch_names = mne_copy.ch_names
        fs = mne_copy.info['sfreq']

        NEW_METHOD = True
        if NEW_METHOD:
            # #
            # import pickle
            # dbg_fname = os.path.join(TEMP_FOLDER, 'first_file_hurst')
            # dbg_calc = False
            # #
            # if dbg_calc:
            #     H_values, LV_values, H_scores, LV_scores = self.calc_hurst_lv_wrapper(mne_copy, show=False, verbose=True, skip_hurst=False)
            #     with open(dbg_fname, 'wb') as fd:
            #         pickle.dump(dict({'H_values': H_values, 'LV_values': LV_values, 'H_scores': H_scores, 'LV_scores': LV_scores}), fd)
            # else:
            #     with open(dbg_fname, 'rb') as fd:
            #         d = pickle.load(fd)
            #     H_values, LV_values, H_scores, LV_scores = d['H_values'], d['LV_values'], d['H_scores'], d['LV_scores']
            # #
            H_values, LV_values, H_scores, LV_scores = self.calc_hurst_lv_wrapper(mne_copy, show=False, verbose=False, skip_hurst=False)
            bad_portions = np.maximum(np.abs(H_scores), np.abs(LV_scores)) > 3
            #num_seg = bad_segment.shape[-1]
            # for i_ch in range(mne_copy.get_data().shape[0]):
            #     if np.any(bad_portions[i_ch]):
            #         plt.plot(mne_copy.get_data()[i_ch])
            #         print('contact', i_ch, '   ', bad_portions[i_ch])
            #         plt.show()
            full_bad_ch = np.all(bad_portions, axis=1)
            partialy_bad_ch = np.any(bad_portions, axis=1)
            exclude_list = np.argwhere(full_bad_ch + partialy_bad_ch).flatten()
            seg_LV_values, seg_LV_scores, boundaries = self.calc_lv_for_segments(mne_copy, exclude_list=exclude_list)
            # plt.plot(LV_scores)
            # plt.show()
            bad_segments = seg_LV_scores > 3
            # now clear the bad segments and check if bad channels are OK again
            v = mne_copy.get_data()
            # fill bad portions with "zeros"
            # #
            # fig, ax = plt.subplots(3, 1)
            # ax[0].plot(v[5])
            # ax[1].plot(v[26])
            # ax[2].plot(v[27])
            # #
            means = np.median(v, axis=1)
            for seg_id in np.argwhere(bad_segments).flatten():
                i_start, i_end = boundaries[np.max(seg_id-1, 0)], boundaries[min(seg_id+1, len(boundaries))]
                for i_ch in exclude_list:
                    v[i_ch, i_start:i_end] = means[i_ch]
            # #
            # ax[0].plot(v[5])
            # ax[1].plot(v[26])
            # ax[2].plot(v[27])
            # #
            bad_channels = []
            for i_ch in exclude_list:
                t_H_values, t_LV_values, _, _ = self.calc_hurst_lv_wrapper(mne_copy, show=False, verbose=True, skip_hurst=False, ch_sel=[i_ch])
                t_H_scores = (t_H_values - H_values.mean()) / (H_values.std() + 1e-12)
                t_LV_scores = (t_LV_values - LV_values.mean()) / (LV_values.std() + 1e-12)
                if logger:
                    # logger.info('\n channel  {} :  '.format(i_ch))
                    # logger.info('old H_scores:    ', H_scores[i_ch])
                    # logger.info('new_H_scores:    ', t_H_scores)
                    # logger.info('old LV_scores:    ', LV_scores[i_ch])
                    # logger.info('new_LV_scores:    ', t_LV_scores)
                    if np.any(np.abs(t_H_scores) > 3) or np.any(np.abs(t_LV_scores) > 3):
                        logger.info('contact {} is bad'.format(i_ch))
                    else:
                        logger.info('return contact {} to list'.format(i_ch))
                if np.any(np.abs(t_H_scores) > 3) or np.any(np.abs(t_LV_scores) > 3):
                    bad_channels.append(i_ch)
            # #
            # plt.show()

            # now insert annotations to the mne object
            mne_copy.info['bads'] = [ch_names[i] for i in bad_channels]
            #
            bad_segments_idxs = np.argwhere(bad_segments).flatten()
            bad_segments_start_idxs = np.maximum(bad_segments_idxs - 1, 0)
            bad_segments_stop_idxs = np.minimum(bad_segments_idxs + 2, boundaries.size - 1)
            bad_segments_starts = boundaries[bad_segments_start_idxs]
            bad_segments_stops = boundaries[bad_segments_stop_idxs]
            bad_segment_durations = bad_segments_stops - bad_segments_starts
            for start, duration in zip(bad_segments_starts, bad_segment_durations):
                description = 'bad sgmnt'
                #event_type = 1000
                onset = start / fs
                duration = duration / fs
                # mne_copy.set_annotations(mne.Annotations(onset, duration, description))
                mne_copy.annotations.append(onset, duration, description)


        return mne_copy





if __name__ == '__main__':


    subject_list = path_utils.get_subject_list()[:2]

    noise_classifier_obj = noise_classifier()

    for subject in subject_list:
        paths = path_utils.get_paths(subject=subject, mode='bipolar')
        for path in paths:
            # try:
            # read .edf file and generate mne_object with annotation of bad chans and segs
            print('\n{}:'.format(path['signals']))
            mne_wrapper = my_mne_wrapper()
            mne_wrapper.read_edf_file(path['signals'])  # ), chanel_groups=electrodes)
            mne_wrapper.preprocess()
            assert mne_wrapper.get_mne().info['sfreq'] == 500
            mne_copy = mne_wrapper.get_mne().copy()
            mne_copy = noise_classifier_obj.mark_bad_chans_and_segs(mne_copy, logger=None)
            print(mne_copy.info['bads'], mne_copy.annotations)

