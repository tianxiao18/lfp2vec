import os
import pickle

import numpy as np
import warnings
import argparse

from scipy.signal import spectrogram

from script.utils import compute_spectrogram_librosa

warnings.filterwarnings('ignore')

def parse_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Parse arguments for LFP data processing.")

    # Add arguments
    parser.add_argument(
        '--chunk_size', 
        type=int, 
        required=True, 
        help="Size of each data chunk to be processed."
    )

    parser.add_argument(
        '--start_time', 
        type=float, 
        required=True, 
        help="Start time for the analysis in seconds."
    )

    parser.add_argument(
        '--n_chunks', 
        type=int, 
        required=True, 
        help="Number of chunks to be processed."
    )

    parser.add_argument(
        '--start_eid', 
        type=int, 
        required=False, 
        default=0,
        help="Start event ID for data processing."
    )

    parser.add_argument(
        '--end_eid', 
        type=int, 
        required=False, 
        default=7,
        help="End event ID for data processing."
    )

    # Parse the arguments
    args = parser.parse_args()

    return args



# def compute_spectrogram(data, sr, time_mul, nperseg=2048):
#     f, t, Sxx = spectrogram(data, sr, nperseg=nperseg)
#     spectrogram_output = 10 * np.log10(Sxx[:500])
#     flattened_spectrogram = spectrogram_output.flatten()

#     return flattened_spectrogram


def load_session(eid):
    return np.load(f'../../data/ibl/{eid}_lfp.npy'), np.load(f'../../data/ibl/{eid}_fs.npy'), np.load(f'../../data/ibl/{eid}_label_lfp.npy', allow_pickle=True)


def create_spectrogram(chunk_size, start_time, n_chunks, start_eid, end_eid):
    print("Job start")
    print(os.getcwd())

    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
    acronyms_arr = np.array(sorted(list(hc_acronyms)))
    acronyms_arr_num = np.arange(len(acronyms_arr))
    acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
    print(acr_dict)
    ## load data
    filtered_sessions = np.load('selected_eids.npy')
    
    print(f"Total number of sessions: {len(filtered_sessions)}")


    # Model execution
    output_path = "../../data/ibl/spectrogram_time"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    for eid in filtered_sessions[start_eid:end_eid]:
        data_list = []
        raw_list = []
        print(f"Processing Session: {eid}")
        region_counts = [0, 0, 0, 0, 0]

        session_data, sampling_rate, labels = load_session(eid)
        ## calculate the time in seconds in the session_data
        time = len(session_data[1]) / sampling_rate
        print(f"Session {eid} has {time} seconds of data")
        data_list = []

        for region in hc_acronyms:
            # chan_ids = np.where(labels == region)[0]
            chan_ids = [i for i, label in enumerate(labels) if region in label]
            regional_lfp_data = session_data[chan_ids]
            lfp_channel_index = chan_ids
            print(f"Region {region} has {len(chan_ids)} channels")

            for i, (chan_id, channel_data) in enumerate(zip(chan_ids, regional_lfp_data)):
                time_mul = chunk_size
                window_size = int(time_mul * sampling_rate)
                start = int(start_time * sampling_rate)
                end = int(start + window_size)
                count = 0
                while count < n_chunks and end < len(channel_data):
                    row = channel_data[start:end]
                    # spect = compute_spectrogram(row, sampling_rate, time_mul)
                    print(f"Shape of row: {row.shape}")
                    z_spect = compute_spectrogram_librosa(row, sampling_rate, num_freq_bins=500, time_bins=16, normalizing='zscore')
                    # db_spect = compute_spectrogram(row, sampling_rate, num_freq_bins=500, time_bins=18, normalizing='db')
                    print(f"Shape of spectrogram: {z_spect.shape}")
                    
                    y = acr_dict[region]
                    ############################# add count index as trial index
                    if not np.any(np.isnan(z_spect)) and not np.any(np.isinf(z_spect)):
                        data_list.append((z_spect, y, count, chan_id))  # , lfp_channel_index[i]))
                        raw_list.append((row, y, count, chan_id))  # , lfp_channel_index[i]))
                        count += 1
                        region_counts[acr_dict[region]] += 1
                    #############################
                    start += window_size
                    end += window_size

        with open(os.path.join(output_path, f"{eid}_data.pickle"), 'wb') as file:
            print(f"Length of data_list: {len(data_list)}")
            pickle.dump(data_list, file)
        with open(os.path.join(output_path, f"{eid}_raw.pickle"), 'wb') as file:
            print(f"Length of raw list: {len(raw_list)}")
            pickle.dump(raw_list, file)
        

        print(region_counts)


# create_spectrogram(chunk_size, start_time, n_chunks)

if __name__ == "__main__":
    args = parse_arguments()
    chunk_size, start_time, n_chunks, start_eid, end_eid = args.chunk_size, args.start_time, args.n_chunks, args.start_eid, args.end_eid
    create_spectrogram(chunk_size, start_time, n_chunks, start_eid, end_eid)