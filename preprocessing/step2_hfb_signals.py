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


def apply_gamma_response(mne, gamma_band=[60, 160], mode='gamma_c'):

    assert mode in ['gamma_c', 'gamma_u'], 'illegal mode {} inside apply_gamma_response'.format(mode)

    if mode == 'gamma_c':
        bw = 20
    if mode == 'gamma_u':
        bw = gamma_band[1] - gamma_band[0]
    fs = new_mne.info['sfreq']
    h = scipy.signal.firls(73, [0, 0.45 * bw, 0.55 * bw, fs / 2], [1, 1, 0, 0], fs=fs)
    # w, H = scipy.signal.freqz(h, 1, fs=fs)
    # plt.semilogy(w, np.real(H * np.conj(H)))
    # plt.grid(True)
    # plt.ylim(1e-8, 2)
    # plt.show()
    boundaries = np.arange(start=gamma_band[0], stop=gamma_band[-1] + bw / 2, step=bw)
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    #band_pwr = np.zeros((v.shape[1], centers.size))
    v = mne.filter(l_freq=10.0, h_freq=None).apply_hilbert().get_data()
    shift_signals = np.zeros((len(centers), v.shape[-1]), dtype=np.complex64)
    gamma_act = np.zeros(v.shape)

    for i_cent, center in enumerate(centers):
        shift_signals[i_cent] = np.exp(-1j * 2 * np.pi * center * np.arange(v.shape[-1]) / fs)

    for i_chan in range(v.shape[0]):
        # fig, ax = plt.subplots(1, 2)
        # print(i_chan)
        for i_cent, center in enumerate(centers):
            x = v[i_chan] * shift_signals[i_cent]
            # w, H = scipy.signal.welch(x, fs=500, return_onesided=False, nfft=2000, nperseg=2000, noverlap=1000, window='hann')
            # ax[0].semilogy(np.fft.fftshift(w), np.fft.fftshift(np.real(H * np.conj(H))))
            y = scipy.signal.filtfilt(h, 1, x)
            # w, H = scipy.signal.welch(y, fs=500, return_onesided=False, nfft=2000, nperseg=2000, noverlap=1000, window='hann')
            # ax[1].semilogy(np.fft.fftshift(w), np.fft.fftshift(np.real(H * np.conj(H))))
            # pwr = np.real(y * np.conj(y))
            # print(center, pwr.mean(), pwr.mean() * (center / min(centers)) ** 2)
            gamma_act[i_chan] += np.abs(y) * (center / min(centers))
        # plt.show()

    mne._data = gamma_act

    return mne




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='gamma_c', help='type of response')
    parser.add_argument('--flow', type=int, default='60', help='lower band frequency')
    parser.add_argument('--fhigh', type=int, default='160', help='higher band frequency')
    parser.add_argument('--partition-id', type=int, default=0, help='The ID of the partition to process (0-indexed).')
    parser.add_argument('--num-partitions', type=int, default=1, help='The total number of partitions.')

    args = parser.parse_args()

    logging.basicConfig(filename=os.path.join(LOG_FOLDER, os.path.basename(__file__).replace('.py', '_{}.log'.format(args.partition_id))), filemode='w', level=logging.DEBUG)

    subject_list = path_utils.get_subject_list()
    subject_list = np.sort(subject_list)[args.partition_id::args.num_partitions]

    FORCE_OVERRIDE = False
    for subject in subject_list:
        logging.info('working on {}'.format(subject) )
        paths = path_utils.get_paths(subject, mode='bipolar')
        for path in paths:
            mode_str = '{}_{}_{}'.format(args.type, args.flow, args.fhigh)
            tgt_fname = path_utils.target_file_name(path['signals'], 'processed', proc_type=mode_str)

            if FORCE_OVERRIDE or (not os.path.isfile(tgt_fname)):
                try:
                    mne_wrapper = my_mne_wrapper()
                    mne_wrapper.read_edf_file(path['signals'])
                    mne_wrapper.preprocess()
                    assert mne_wrapper.get_mne().info['sfreq'] == 500

                    new_mne = mne_wrapper.get_mne().copy()
                    new_mne = apply_gamma_response(new_mne, gamma_band=[args.flow, args.fhigh], mode=args.type)
                    new_mne.resample(sfreq=100)

                    annot_fname = path_utils.target_file_name(path['signals'], 'annot')
                    with open(annot_fname, 'rb') as fd:
                        annotations = pickle.load(fd)
                    new_mne.info['bads'] = annotations['bads']
                    new_mne.set_annotations(annotations['annotations'])

                    os.makedirs(os.path.dirname(tgt_fname), exist_ok=True)
                    new_mne.save(tgt_fname, overwrite=True)
                    logging.info('file {} has been written'.format(tgt_fname))


                    # mne_wrapper.get_mne().info['bads'] = annotations['bads']
                    # mne_wrapper.get_mne().set_annotations(annotations['annotations'])
                    # mne_wrapper.get_mne().plot()
                    # new_mne.plot()
                    # plt.show()
                except:
                    logging.warning('FAILED to generate {}'.format(tgt_fname))
            else:
                logging.info('already exists: {}'.format(tgt_fname))

