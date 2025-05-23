import warnings
import gc

from collections import defaultdict
from tqdm import tqdm
from script.utils import *
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

warnings.filterwarnings('ignore')


def raw_data_creator(chunk_size, start_time, n_chunks, data="Allen"):
    print("Job start")
    print(os.getcwd())
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Visual_Cortex'}
    acronyms_arr = np.array(sorted(list(hc_acronyms)))
    acronyms_arr_num = np.arange(len(acronyms_arr))
    acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
    print(acr_dict)

    output_path = os.path.join("raw", data)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if data == "Allen":
        output_path = f'../../script/raw/{data}'
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
        # mask = sessions_list.index.isin([773418906, 781842082])
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
            data_list = []
            region_counts = [0, 0, 0, 0, 0]
            print(f"Processing Session: {session}")
            # Get the NWB file for the session
            session_data = data_cache.get_session_data(int(session))

            for probe in tqdm(probe_files):
                # Get the NWB file for the probe
                try:
                    lfp_data = session_data.get_lfp(int(probe))
                except Exception as e:
                    continue

                sampling_rate = session_data.probes.loc[int(probe)].lfp_sampling_rate

                for region in hc_acronyms:
                    chan_ids = (session_data.channels[(session_data.channels.probe_id == int(probe)) &
                                                      (session_data.channels.ecephys_structure_acronym.isin(
                                                          [region]))].index.values)
                    if len(chan_ids) == 0:
                        continue

                    regional_lfp_data = lfp_data.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))

                    for i, channel_data in enumerate(regional_lfp_data.data.T):
                        time_mul = chunk_size
                        window_size = int(time_mul * sampling_rate)
                        start = int(start_time * sampling_rate)
                        end = int(start + window_size)
                        count = 0
                        while count < n_chunks and end < len(channel_data):
                            row = channel_data[start:end]
                            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                                base_rms_mini = calculate_rms(row)
                                pow_whole_mini, rms_whole_mini = calculate_power(row, sampling_rate)
                                pow_delta_mini, rms_delta_mini = calculate_power(row, sampling_rate, (0, 4))
                                pow_theta_mini, rms_theta_mini = calculate_power(row, sampling_rate, (4, 8))
                                pow_alpha_mini, rms_alpha_mini = calculate_power(row, sampling_rate, (8, 12))
                                pow_beta_mini, rms_beta_mini = calculate_power(row, sampling_rate, (12, 30))
                                pow_gamma_mini, rms_gamma_mini = calculate_power(row, sampling_rate, (30, -1))

                                # Append data to the list
                                # (Data, Label, Trial index, Channel index)
                                data_list.append(
                                    ([base_rms_mini, pow_whole_mini, pow_delta_mini, pow_theta_mini, pow_alpha_mini,
                                      pow_beta_mini, pow_gamma_mini, rms_whole_mini, rms_delta_mini, rms_theta_mini,
                                      rms_alpha_mini, rms_beta_mini, rms_gamma_mini], acr_dict[region], count, f"{probe}_{i}")
                                )

                                start += window_size
                                end += window_size
                                count += 1

                                region_counts[acr_dict[region]] += 1

                # Add Visual Cortex data only if there is data from Hippocampus regions
                if len(data_list) != 0:
                    vis_acr = {'VIS', 'VISal', 'VISam', 'VISl', 'VISli', 'VISmma', 'VISmmp', 'VISp', 'VISpm', 'VISrl'}
                    chan_ids = (session_data.channels[(session_data.channels.probe_id == int(probe)) &
                                                      (session_data.channels.ecephys_structure_acronym.isin(
                                                          vis_acr))].index.values)
                    if len(chan_ids) == 0:
                        continue
                    regional_lfp_data = lfp_data.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))

                    for i, channel_data in enumerate(regional_lfp_data.data.T):
                        time_mul = chunk_size
                        window_size = int(time_mul * sampling_rate)
                        start = int(start_time * sampling_rate)
                        end = int(start + window_size)
                        count = 0
                        while count < n_chunks and end < len(channel_data):
                            row = channel_data[start:end]
                            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                                base_rms_mini = calculate_rms(row)
                                pow_whole_mini, rms_whole_mini = calculate_power(row, sampling_rate)
                                pow_delta_mini, rms_delta_mini = calculate_power(row, sampling_rate, (0, 4))
                                pow_theta_mini, rms_theta_mini = calculate_power(row, sampling_rate, (4, 8))
                                pow_alpha_mini, rms_alpha_mini = calculate_power(row, sampling_rate, (8, 12))
                                pow_beta_mini, rms_beta_mini = calculate_power(row, sampling_rate, (12, 30))
                                pow_gamma_mini, rms_gamma_mini = calculate_power(row, sampling_rate, (30, -1))

                                # Append data to the list
                                # (Data, Label, Trial index, Channel index)
                                data_list.append(
                                    ([base_rms_mini, pow_whole_mini, pow_delta_mini, pow_theta_mini, pow_alpha_mini,
                                      pow_beta_mini, pow_gamma_mini, rms_whole_mini, rms_delta_mini, rms_theta_mini,
                                      rms_alpha_mini, rms_beta_mini, rms_gamma_mini], 4, count, f"{probe}_{i}")
                                )

                                start += window_size
                                end += window_size
                                count += 1

                                region_counts[4] += 1

            del lfp_data
            gc.collect()
            print(f"Data size: {len(data_list)}")
            print(region_counts)
            # with open(os.path.join(output_path, f"{session}_data.pickle"), 'wb') as file:
            #     print(f"Length of data_list: {len(data_list)}")
            #     pickle.dump(data_list, file)

        del session_data
        gc.collect()


# raw_data_creator(3, 200, 75)
