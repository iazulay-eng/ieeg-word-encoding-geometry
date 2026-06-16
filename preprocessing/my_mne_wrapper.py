import numpy as np
import mne


class my_mne_wrapper:

    def __init__(self):

        super().__init__()


    def _bipolar_montage_from_monopolar(self, chanel_groups, edf_fname, closed_groups=True):

        temp_mne = mne.io.read_raw_edf(edf_fname, preload=False)
        bipolar_chanel_groups = dict()
        for region in chanel_groups:
            bipolar_chanel_groups[region] = []
            chanel_gruop_electrodes = [e['name'] for e in chanel_groups[region]]

            for bi_channel in temp_mne.ch_names:
                ch1, ch2 = bi_channel.split('-')
                ch1_ok, ch2_ok = ch1 in chanel_gruop_electrodes, ch2 in chanel_gruop_electrodes
                ok = ch1_ok and ch2_ok if closed_groups else ch1_ok or ch2_ok
                if ok:
                    bipolar_chanel_groups[region].append(bi_channel)

        return bipolar_chanel_groups

    def read_edf_file(self, fname, chanel_groups=None):

        self.is_monopolar = fname.find('monopolar') > -1
        self.is_bipolar = fname.find('bipolar') > -1
        assert self.is_monopolar ^ self.is_bipolar


        if chanel_groups is None:
            self.mne = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        else:
            self.monopolar_chanel_gruops = dict()
            for r in chanel_groups:
                self.monopolar_chanel_gruops[r] = [e['name'] for e in chanel_groups[r]]

            if self.is_monopolar:
                include_list = [e['name'] for r in chanel_groups for e in chanel_groups[r]]
            if self.is_bipolar:
                self.bipolar_chanel_groups = self._bipolar_montage_from_monopolar(chanel_groups=chanel_groups, edf_fname=fname)
                include_list = [e for r in chanel_groups for e in self.bipolar_chanel_groups[r]]

            if len(include_list) > 0:
                self.mne = mne.io.read_raw_edf(fname, preload=True, include=include_list, verbose=False)
            else:
                self.exceptions = 'no contacts'
                return

        self.original_sfreq = self.mne.info['sfreq']
        if self.original_sfreq != 500:
            self.mne.resample(500)



    def set_events(self, events, event_glossary):

        annot = mne.annotations_from_events(events=events, sfreq=self.mne.info['sfreq'], event_desc=event_glossary)
        self.mne.set_annotations(annot)


    def get_mne(self, region_subset=None):

        if region_subset is None:
            return self.mne
        else:
            assert False, 'not implemented yet'


    def preprocess(self, powerline=60, passband=None):

        if powerline is not None:
            frequencies = np.arange(start=powerline, stop=self.mne.info['sfreq'] / 2, step=powerline)
            self.mne.notch_filter(freqs=frequencies, filter_length='auto', notch_widths=1, trans_bandwidth=0.3, method='fir', phase='zero', verbose=False)

        if passband is not None:
            self.mne.filter(l_freq=min(passband), h_freq=max(passband), phase='zero', verbose=False)



if __name__ == '__main__':

    from my_montage_reader import my_montage_reader

    montage_fname = 'E:/ds004789-download/sub-R1001P/ses-0/ieeg/sub-R1001P_ses-0_task-FR1_space-MNI152NLin6ASym_electrodes.tsv'
    montage = my_montage_reader(fname=montage_fname)
    region_list = ['entorihinal', 'cuneus', 'fusiform', 'lateraloccipital', 'lingual', 'precuneus', 'superiorpariental']
    hemisphere_sel = ['LR', 'both', 'LR', 'LR', 'both', 'both', 'LR']
    electrode_list = montage.get_electrode_list_by_region(region_list=region_list, hemisphere_sel=hemisphere_sel)

    edf_fname = 'E:/ds004789-download/sub-R1001P/ses-0/ieeg/sub-R1001P_ses-0_task-FR1_acq-monopolar_ieeg.edf'
    signals = my_mne_wrapper()
    signals.read_edf_file(fname=edf_fname, chanel_groups=electrode_list)
    print('here')
    import matplotlib.pyplot as plt
    scale =  dict(mag=1e-8, grad=4e-10, eeg=20e-4, eog=150e-5, ecg=5e-5,emg=1e-2, ref_meg=1e-11, misc=1e-2, stim=1,resp=1, chpi=1e-4, whitened=1e2)
    signals.mne.plot(scalings=scale)
    plt.show()

    edf_fname = 'E:/ds004789-download/sub-R1001P/ses-0/ieeg/sub-R1001P_ses-0_task-FR1_acq-bipolar_ieeg.edf'
    signals.read_edf_file(fname=edf_fname, chanel_groups=electrode_list)
    print('here')
