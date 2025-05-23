# from spikeglx import Reader
import spikeglx
# from ibllib.plots import Density
# import matplotlib.pyplot as pl
from ibldsp.voltage import destripe
from brainbox.io.one import SpikeSortingLoader

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from ibldsp.voltage import destripe_lfp

# import matplotlib.pyplot as plt
# import scipy.signal
# from brainbox.ephys_plots import plot_brain_regions
# from ibllib.plots import Density
# from ibldsp.voltage import destripe
import numpy as np

# from sklearn.model_selection import train_test_split
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report, accuracy_score
import seaborn as sns
# import argparse
from tqdm import tqdm
import os

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
# one = ONE(password='international')
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', cache_dir='../../data/ibl/cache')

# eids = np.load('selected_eids.npy')
# eids = np.load("/scratch/cl7201/ibl/dataset/CA1CA2CA3/CA1CA2CA3_eid.npy")
eids = pd.read_csv("/scratch/th3129/region_decoding/data/ibl/re_eids.csv")["eid"]

def download_lfp():
    target = ['CA1', 'CA2', 'CA3', 'DG', 'VIS']

    for eid in tqdm(eids): 
        # if exist in "../../data/ibl/", skip
        datasets = one.list_datasets(eid, collection='alf/probe*')
        probe_labels = set(d.split('/')[1] for d in datasets)  # List the insertions
        # You can find full details of a session's insertions using the following database query:
        data_list = []
        label_list = []
        fs_list = []
        for probe in probe_labels:
            if os.path.exists(f'/vast/th3129/data/ibl_new/{eid}_{probe}_lfp.npy'):
                print(f"skip {eid}")
                continue
                
            ssl = SpikeSortingLoader(eid=eid, one=one, pname=probe)
            
            # The channels information is contained in a dict table / dataframe
            channels = ssl.load_channels()
            labels = channels['acronym']
            dsets = one.list_datasets(eid, collection=f'raw_ephys_data/{probe}', filename='*.lf.*')
            data_files, _ = one.load_datasets(eid, dsets, download_only=True)
            bin_file = next(df for df in data_files if df.suffix == '.cbin')
            sr = spikeglx.Reader(bin_file)
            
            
            print("unique labels\n"+ str(np.unique(labels)))
            # check if unique labels contain any target, if so ,continue, if not, break it

            if not any([t in labels for t in target]):
                print("skip")
                continue

            # print("label shape\n"+ str(labels.shape))
            # print("what is sr-lf\n"+ str(type(sr)))

            # print("sr_lf shape\n"+ str(sr.shape))
            print(f"streaming length {int(sr.fs*500)}")
            print("sampling rate " + str(sr.fs))
            data = sr.read_samples(first_sample=0, last_sample=int(sr.fs*500))  # Read 500 seconds of samples
            destriped_lfp = destripe_lfp(data[0][:,:-sr.nsync].T, fs=sr.fs, channel_labels=True)
            print(destriped_lfp.shape)
            # printw("data shape\n"+ str(data.shape))
            # print("data shape\n"+ str(data[0][:,:-sr.nsync].shape))
            #integrate data to data list
            #### n channel * samples
            # data_list.append(data[0][:,:-sr.nsync].T)
            # label_list.append(labels)
            # fs_list.append(sr.fs)
        # if len(data_list) > 1:
            # concatenate labels to np array
            # label_list = np.concatenate(label_list,axis=0)
            np.save(f'/vast/th3129/data/ibl_new/{eid}_{probe}_label_lfp.npy', labels)
            # save data
            # data_list = np.concatenate(data_list,axis=0)
            np.save(f'/vast/th3129/data/ibl_new/{eid}_{probe}_lfp.npy', destriped_lfp)
            # save sampling rate
            # fs_list = np.concatenate(fs_list)
            # print(fs_list)
            np.save(f'/vast/th3129/data/ibl_new/{eid}_{probe}_fs.npy', sr.fs)
            print(f"Finished downloading LFP for hip regions in {eid}, shapes:{data[0][:,:-sr.nsync].T.shape}")
        
    return
    

def download_spike():
    print("unique eids", len(eids))
    for i,eid in tqdm(enumerate(eids)):
        print(i,eid)
        datasets = one.list_datasets(eid, collection='alf/probe*')
        probe_labels = set(d.split('/')[1] for d in datasets)  # List the insertions

        for probe in probe_labels:
            ssl = SpikeSortingLoader(eid=eid, one=one, pname=probe)
            channels = ssl.load_channels()
            spikes, clusters, channels = ssl.load_spike_sorting()
            clusters = ssl.merge_clusters(spikes, clusters, channels)

            spike_time = {}

            for j in range(spikes['times'].shape[0]):
                idx = spikes['clusters'][j]
                
                if idx not in clusters['cluster_id']:
                    continue
                temp_idx = np.array(clusters['cluster_id'] == idx).nonzero()[0][0]
                
                channel = clusters['channels'][temp_idx]
                
                if channel not in spike_time:
                    spike_time[channel] = []
                
                spike_time[channel].append(spikes['times'][j])
            label = channels['acronym']
           
            np.save(f'../../data/{eid}_{probe}_label.npy', label)
            np.save(f'../../data/{eid}_{probe}_spike_time.npy', spike_time)

    return

def download_waveform():
    for eid in tqdm(eids):
        datasets = one.list_datasets(eid, collection='alf/probe*')
        probe_labels = set(d.split('/')[1] for d in datasets)  # List the insertions

        for probe in probe_labels:
            ssl = SpikeSortingLoader(eid=eid, one=one, pname=probe)
            channels = ssl.load_channels()
            spikes, clusters, channels = ssl.load_spike_sorting()
            clusters = ssl.merge_clusters(spikes, clusters, channels)
            waveforms = ssl.load_waveforms()
            
            # Print available fields in waveforms
            print(f"Available fields in waveforms: {waveforms.keys()}")
            
            # only save template
            waveforms_templates = waveforms['templates']
            waveforms_traces = waveforms['traces']
            # print shape
            print(f"waveform shape: {waveforms_templates.shape}")
            print(f"waveform traces shape: {waveforms_traces.shape}")
            np.save(f'../../data/ibl/{eid}_{probe}_waveform_templates.npy', waveforms_templates)
            np.save(f'../../data/ibl/{eid}_{probe}_waveform_traces.npy', waveforms_traces)
            print(f"Finished downloading waveform for {eid} {probe}")
            break
        break

    return


def combine_probe():
    # search files under '../../data2/ibl/'
    # combine all the probe data into one file
    # save the combined data into '../../data/ibl/{eid}_lfp.npy'
    pickle_path = '/vast/th3129/data/ibl_new/'
    probe_files = os.listdir(pickle_path)
    for eid in eids:
        if os.path.exists(f'/vast/th3129/data/ibl_new/{eid}_lfp.npy'):
            print(f"skip {eid}")
            continue
        data_list = []
        label_list = []
        fs_list = []
        for probe in probe_files:
            if eid in probe:
                if 'label' in probe:
                    label = np.load(pickle_path + probe, allow_pickle=True)
                    label_list.append(label)
                elif 'lfp' in probe:
                    data = np.load(pickle_path + probe)
                    data_list.append(data)
                elif 'fs' in probe:
                    fs = np.load(pickle_path + probe)
                    fs_list.append(fs)
        # if datalist has different length, cut the data to the same length as the shortest one
        min_length = min([data.shape[1] for data in data_list])
        data_list = [data[:,:min_length] for data in data_list]
        print(f"Combined data shape: {np.concatenate(data_list, axis=0).shape}")
        data_list = np.concatenate(data_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        np.save(f'/vast/th3129/data/ibl_new/{eid}_lfp.npy', data_list)
        np.save(f'/vast/th3129/data/ibl_new/{eid}_label_lfp.npy', label_list)
        if len(fs_list) > 1:
            assert fs_list[0] == fs_list[1]
            fs_list = fs_list[0]
        np.save(f'/vast/th3129/data/ibl_new/{eid}_fs.npy', fs_list)


    return 
if __name__ == "__main__":
    download_lfp()
    # download_spike()
    # combine_probe()
    # download_waveform()