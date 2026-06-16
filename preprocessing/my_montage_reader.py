import mne.viz
import numpy as np
import pandas as pd


class my_montage_reader:

    def __init__(self, fname=None):

        if fname is not None:
            self.read_montage_file(fname=fname)

    def read_montage_file(self, fname):

        assert fname[-4:] == '.tsv'
        self.df = pd.read_csv(fname, delimiter='\t')


    def get_electrode_list_by_region(self, region_list, hemisphere_sel=None):

        # defualt hemisphere selectivity: no selectivity
        if hemisphere_sel is None:
            hemisphere_sel_list = ['both' for region in region_list]
        else:
            hemisphere_sel_list = hemisphere_sel

        electrode_list = dict()

        for i_region, region in enumerate(region_list):
            if hemisphere_sel_list[i_region] == 'both':
                electrode_list[region] = []
            else:
                if hemisphere_sel_list[i_region].find('L') > -1:
                    electrode_list[region + '-L'] = []
                if hemisphere_sel_list[i_region].find('R') > -1:
                    electrode_list[region + '-R'] = []

        for i in range(self.df.shape[0]):
            line = self.df.iloc[i]
            for i_region, region in enumerate(region_list):
                if ('ind.region' in line.keys()) and (region == line['ind.region']):

                    admit = False
                    if hemisphere_sel_list[i_region] == 'both':
                        admit, key = True, region
                    try:
                        if hemisphere_sel_list[i_region].find(line['hemisphere']) > -1:
                            admit, key = True, region + '-' + line['hemisphere']
                    except:
                        pass # probably line['hemisphere'] is corrupted
                    if admit:
                        electrode_list[key].append(dict({'name': line['name'], 'coords': np.array([line['x'], line['y'], line['z']]).reshape(3, 1), 'hemisphere': line['hemisphere']}))

        return electrode_list


if __name__ == '__main__':

    filename = 'E:/ds004789-download/sub-R1001P/ses-0/ieeg/sub-R1001P_ses-0_task-FR1_space-MNI152NLin6ASym_electrodes.tsv'
    montage = my_montage_reader(fname=filename)
    list = montage.get_electrode_list_by_region(['xxx', 'yyy', 'fusiform', 'lingual'], hemisphere_sel=['both', 'L', 'LR', 'both'])
    # now try a preliminary relevant list
    region_list = ['entorihinal', 'cuneus', 'fusiform', 'lateraloccipital', 'lingual', 'precuneus', 'superiorpariental']
    hemisphere_sel = ['LR', 'both', 'LR', 'LR', 'both', 'both', 'LR']
    list = montage.get_electrode_list_by_region(region_list=region_list, hemisphere_sel=hemisphere_sel)
    print(list)

    import matplotlib.pyplot as plt
    # electrode_coords = np.concatenate((np.atleast_2d(montage.df.x.values), np.atleast_2d(montage.df.y.values), np.atleast_2d(montage.df.z.values))).T
    # mne.utils.set_config('SUBJECTS_DIR', 'E:/freesurfer/7.4.1/subjects')
    # brain = mne.viz.Brain('fsaverage', hemi='both', surf='pial', views='lateral', background='white')
    # brain.add_foci(electrode_coords, hemi='vol', color='red', scale_factor=0.2)
    # #brain.add_foci(electrode_coords, hemi='lh', color='green', scale_factor=0.2)
    # labels = mne.read_labels_from_annot('fsaverage', parc='aparc.a2009s')
    # for label in labels:
    #     brain.add_label(label=label)
    # brain.show()
    # fig, ax = plt.subplots(1, 1)
    # plt.show(block=True)

    from path_utils import get_paths, get_subject_list
    base_folder = 'E:/ds004789-download'
    regions, sides = ['fusiform', 'inferiortemporal', 'lateraloccipital', 'lingual'], ['LR', 'LR', 'LR', 'both']
    electrode_coords_dict = dict()
    subject_list = ['sub-R1001P', 'sub-R1002P', 'sub-R1003P', 'sub-R1006P', 'sub-R1010J', 'sub-R1034D', 'sub-R1054J',
                    'sub-R1059J', 'sub-R1060M', 'sub-R1065J',
                    'sub-R1066P', 'sub-R1067P', 'sub-R1068J', 'sub-R1070T', 'sub-R1077T', 'sub-R1080E', 'sub-R1083J',
                    'sub-R1092J', 'sub-R1094T', 'sub-R1100D',
                    'sub-R1106M', 'sub-R1108J', 'sub-R1111M', 'sub-R1112M', 'sub-R1113T', 'sub-R1118N', 'sub-R1123C',
                    'sub-R1124J', 'sub-R1125T', 'sub-R1134T',
                    'sub-R1136N', 'sub-R1145J', 'sub-R1147P', 'sub-R1153T', 'sub-R1154D', 'sub-R1158T', 'sub-R1161E',
                    'sub-R1163T', 'sub-R1167M', 'sub-R1168T',
                    'sub-R1170J', 'sub-R1171M', 'sub-R1172E', 'sub-R1174T', 'sub-R1177M', 'sub-R1195E', 'sub-R1196N',
                    'sub-R1201P', 'sub-R1202M', 'sub-R1204T',
                    'sub-R1215M', 'sub-R1226D', 'sub-R1234D', 'sub-R1240T', 'sub-R1243T', 'sub-R1281E', 'sub-R1283T',
                    'sub-R1293P', 'sub-R1297T', 'sub-R1299T',
                    'sub-R1302M', 'sub-R1308T', 'sub-R1309M', 'sub-R1310J', 'sub-R1311T', 'sub-R1315T', 'sub-R1316T',
                    'sub-R1317D', 'sub-R1323T', 'sub-R1325C',
                    'sub-R1328E', 'sub-R1331T', 'sub-R1334T', 'sub-R1336T', 'sub-R1337E', 'sub-R1338T', 'sub-R1341T',
                    'sub-R1345D', 'sub-R1346T', 'sub-R1350D',
                    'sub-R1354E', 'sub-R1355T', 'sub-R1361C', 'sub-R1363T', 'sub-R1367D', 'sub-R1374T', 'sub-R1377M',
                    'sub-R1378T', 'sub-R1379E', 'sub-R1385E',
                    'sub-R1386T', 'sub-R1387E', 'sub-R1391T', 'sub-R1394E', 'sub-R1396T', 'sub-R1405E', 'sub-R1415T',
                    'sub-R1416T', 'sub-R1420T', 'sub-R1422T',
                    'sub-R1425D', 'sub-R1427T', 'sub-R1443D', 'sub-R1449T', 'sub-R1463E', 'sub-R1542J']
    for subject in subject_list:
        paths = get_paths(base_folder=base_folder, subject=subject)
        if len(paths) == 0:
            continue
        montage = my_montage_reader(fname=paths[0]['electrodes'])
        print(subject)
        list = montage.get_electrode_list_by_region(region_list=regions, hemisphere_sel=sides)
        for key in list:
            if not (key in electrode_coords_dict):
                electrode_coords_dict[key] = np.zeros((0, 3))
            if len(list[key]) > 0:
                electrode_coords_dict[key] = np.concatenate((electrode_coords_dict[key], np.array([e['coords'] for e in list[key]])[:, :, 0]), axis=0)

    num_grps = len(electrode_coords_dict)
    clrs = np.zeros((num_grps, 3))
    for i in range(num_grps):
        clrs[i] = np.array((1 - i/num_grps, 0.5*(i % 2), i/num_grps))

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 4))
    mne.utils.set_config('SUBJECTS_DIR', 'E:/freesurfer/7.4.1/subjects')
    brain = mne.viz.Brain('fsaverage', hemi='both', surf='pial', views='lateral', background='white')
    for i, key in enumerate(electrode_coords_dict):
        brain.add_foci(electrode_coords_dict[key], hemi='vol', color=clrs[i], scale_factor=0.2)
        ax.text(0.1, 0.1 + i/10, key, c=clrs[i], fontsize=12)
        #print(electrode_coords_dict[key])
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc.a2009s')
    for label in labels:
        brain.add_label(label=label)
    brain.show()
    #ax.legend()
    plt.show(block=True)



