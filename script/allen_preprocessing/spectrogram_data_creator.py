import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import librosa
import numpy as np
import warnings
import gc

from collections import defaultdict
from script.utils import *
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from tqdm import tqdm

warnings.filterwarnings('ignore')


def create_spectrogram(chunk_size, start_time, n_chunks, data="Allen"):
    print("Job start")
    print(os.getcwd())
    flag = True

    # Model execution
    output_path = os.path.join("spectrogram", data)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Visual_Cortex'}
    acronyms_arr = np.array(sorted(list(hc_acronyms)))
    acronyms_arr_num = np.arange(len(acronyms_arr))
    acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
    print(acr_dict)

    if data == "Allen":
        os.chdir('../../data/Allen')
        output_path = f'../../script/spectrogram/{data}'
        current_dir = os.getcwd()
        manifest_path = 'data'
        manifest_path = os.path.join(manifest_path, "manifest.json")
        if not os.path.exists(manifest_path) and ~current_dir.endswith('Allen'):
            os.chdir('../data/Allen')

        if not os.path.exists(manifest_path):
            os.makedirs(manifest_path)

        data_cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

        sessions_list = data_cache.get_session_table()
        mask = (sessions_list['ecephys_structure_acronyms']
                .apply(lambda acronyms: bool(hc_acronyms.intersection(set(acronyms)))))
        filtered_sessions = sessions_list[mask]
        print(f"Total number of sessions: {len(filtered_sessions)}")

        probe_files_map = defaultdict(list)

        for root, dirs, files in os.walk(os.getcwd()):
            if os.path.basename(root).startswith('session_'):
                session_folder_name = os.path.basename(root).replace('session_', '')
                for file in files:
                    if file.startswith('probe_') and file.endswith('_lfp.nwb'):
                        (probe_files_map[session_folder_name]
                         .append(file.replace('_lfp.nwb', '').replace('probe_', '')))

        print(probe_files_map)

        for session, probe_files in probe_files_map.items():
            region_counts = [0, 0, 0, 0, 0]
            data_list = []
            raw_list = []
            print(f"Processing Session: {session}")
            # Get the NWB file for the session
            session_data = data_cache.get_session_data(int(session))
            temp_count = 0

            for probe in tqdm(probe_files):
                # Get the NWB file for the probe
                try:
                    lfp_data = session_data.get_lfp(int(probe))
                except Exception as e:
                    continue

                sampling_rate = session_data.probes.loc[int(probe)].lfp_sampling_rate  # 1249.998518276065
                min_chan, max_chan = 0, 0

                for region in hc_acronyms:
                    chan_ids = (session_data.channels[(session_data.channels.probe_id == int(probe)) &
                                                      (session_data.channels.ecephys_structure_acronym.isin(
                                                          [region]))].index.values)
                    if len(chan_ids) == 0:
                        continue

                    regional_lfp_data = lfp_data.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))
                    lfp_channel_index = regional_lfp_data.channel
                    min_chan, max_chan = min(np.min(chan_ids), min_chan), max(np.max(chan_ids), max_chan)

                    for i, channel_data in enumerate(regional_lfp_data.data.T):
                        time_mul = chunk_size
                        window_size = int(time_mul * sampling_rate)
                        start = int(start_time * sampling_rate)
                        end = int(start + window_size)
                        count = 0
                        while count < n_chunks and end < len(channel_data):
                            row = channel_data[start:end]

                            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                                z_spect = compute_spectrogram_librosa(row, sampling_rate, 500, 21)
                                y = acr_dict[region]
                                if flag == True:
                                    print(z_spect.shape)
                                    flag = False

                                if not np.any(np.isnan(z_spect)) and not np.any(np.isinf(z_spect)):
                                    data_list.append(
                                        (z_spect, y, count, f"{probe}_{lfp_channel_index[i].item()}"))  # , lfp_channel_index[i]))
                                    raw_list.append((row, y, count, f"{probe}_{lfp_channel_index[i].item()}"))  # , lfp_channel_index[i]))
                                    count += 1
                                    region_counts[acr_dict[region]] += 1

                            start += window_size
                            end += window_size

                # Add Visual Cortex data only if there is data from Hippocampus regions
                if len(data_list) != 0:
                    vis_acr = {'VIS', 'VISal', 'VISam', 'VISl', 'VISli', 'VISmma', 'VISmmp', 'VISp', 'VISpm', 'VISrl'}
                    chan_ids = (session_data.channels[(session_data.channels.probe_id == int(probe)) &
                                                      (session_data.channels.ecephys_structure_acronym.isin(
                                                          vis_acr))].index.values)
                    if len(chan_ids) == 0:
                        continue
                    # Count chan_ids that are in the range of min_chan and max_chan
                    temp = chan_ids[(chan_ids >= min_chan) & (chan_ids <= max_chan)]
                    temp_count += len(temp)
                    regional_lfp_data = lfp_data.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))
                    lfp_channel_index = regional_lfp_data.channel

                    for i, channel_data in enumerate(regional_lfp_data.data.T):
                        time_mul = chunk_size
                        window_size = int(time_mul * sampling_rate)
                        start = int(start_time * sampling_rate)
                        end = int(start + window_size)
                        count = 0
                        while count < n_chunks and end < len(channel_data):
                            row = channel_data[start:end]

                            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                                z_spect = compute_spectrogram_librosa(row, sampling_rate, 500, 21)
                                y = 4

                                if not np.any(np.isnan(z_spect)) and not np.any(np.isinf(z_spect)):
                                    data_list.append(
                                        (z_spect, y, count, f"{probe}_{lfp_channel_index[i].item()}"))  # , lfp_channel_index[i]))
                                    raw_list.append((row, y, count, f"{probe}_{lfp_channel_index[i].item()}"))  # , lfp_channel_index[i]))
                                    count += 1
                                    region_counts[4] += 1

                            start += window_size
                            end += window_size

                # del lfp_data
                # gc.collect()

            if len(data_list) != 0:
                with open(os.path.join(output_path, f"{session}_data.pickle"), 'wb') as file:
                    print(f"Length of data_list: {len(data_list)}")
                    print(f"Intersecting channels: {temp_count}")
                    pickle.dump(data_list, file)
                with open(os.path.join(output_path, f"{session}_raw.pickle"), 'wb') as file:
                    print(f"Length of raw list: {len(raw_list)}")
                    pickle.dump(raw_list, file)

                print(region_counts)
            else:
                print(f"No data for session: {session}")
            del session_data
            gc.collect()


create_spectrogram(3, 200, 100)
