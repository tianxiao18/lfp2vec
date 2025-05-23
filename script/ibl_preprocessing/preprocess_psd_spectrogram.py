import os
import pickle

import numpy as np
import warnings
import argparse

from scipy.signal import spectrogram
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from script.utils import compute_spectrogram_librosa, calculate_rms, calculate_power

import scipy.signal as signal
import time
warnings.filterwarnings('ignore')
from scipy.io import savemat

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


def preprocess_signal(data, original_sampling_rate, target_sampling_rate, spatial_downsample_factor=4, low_pass=False):
    """
    Preprocesses the signal by downsampling, filtering, and re-referencing.

    Parameters:
    - data: 2D numpy array (channels x samples)
    - original_sampling_rate: Original sampling rate in Hz
    - target_sampling_rate: Target sampling rate in Hz
    - spatial_downsample_factor: Factor to downsample channels

    Returns:
    - preprocessed_data: 2D numpy array of preprocessed data
    """

    start_time = time.time()
    # Temporal Downsampling
    downsample_factor = int(original_sampling_rate / target_sampling_rate)
    if downsample_factor > 1:
        # Anti-aliasing filter
        nyquist_original = 0.5 * original_sampling_rate
        cutoff = 0.5 * target_sampling_rate  # To preserve frequencies below Nyquist after downsampling
        sos = signal.butter(4, cutoff / nyquist_original, btype='low', output='sos')
        data = signal.sosfiltfilt(sos, data, axis=1)
        data = data[:, ::downsample_factor]
    else:
        print("No temporal downsampling applied.")

    # # Spatial Downsampling
    # data = data[::spatial_downsample_factor, :]

    # Low-pass Filtering at 500 Hz (if applicable)
    if low_pass:
        nyquist_new = 0.5 * target_sampling_rate
        sos = signal.butter(2, 500 / nyquist_new, btype='lowpass', output='sos')
        data = signal.sosfiltfilt(sos, data, axis=1)

    # High-pass Filtering at 0.1 Hz
    nyquist_new = 0.5 * target_sampling_rate
    sos = signal.butter(2, 0.1 / nyquist_new, btype='highpass', output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)

    # Re-referencing to common average (assuming no external reference channels)
    # reference = np.mean(data, axis=0)
    # data = data - reference
    print(f"Preprocessing took {time.time() - start_time} seconds.")

    return data
# def compute_spectrogram(data, sr, time_mul, nperseg=2048):
#     f, t, Sxx = spectrogram(data, sr, nperseg=nperseg)
#     spectrogram_output = 10 * np.log10(Sxx[:500])
#     flattened_spectrogram = spectrogram_output.flatten()

#     return flattened_spectrogram


def load_session(eid):
    return np.load(f'/vast/th3129/data/ibl_new/{eid}_lfp.npy'), np.load(f'/vast/th3129/data/ibl_new/{eid}_fs.npy'), np.load(f'/vast/th3129/data/ibl_new/{eid}_label_lfp.npy', allow_pickle=True)
    # return np.load(f'/vast/th3129/data/ibl/{eid}_lfp.npy'), np.load(f'/vast/th3129/data/ibl/{eid}_fs.npy'), np.load(f'/vast/th3129/data/ibl/{eid}_label_lfp.npy', allow_pickle=True)


def create_spectrogram(chunk_size, start_time, n_chunks, start_eid, end_eid):
    print("Job start")
    print(os.getcwd())

    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
    acronyms_arr = np.array(sorted(list(hc_acronyms)))
    acronyms_arr_num = np.arange(len(acronyms_arr))
    acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
    print(acr_dict)
    ## load data
    # filtered_sessions = np.load('/scratch/cl7201/ibl/dataset/CA1CA2CA3/CA1CA2CA3_eid.npy')
    input_path = '/vast/th3129/data/ibl_new'
    filtered_sessions = sorted(list(set(
                            '_'.join(p.split('_')[:2]) for p in os.listdir(input_path)
                            if not os.path.isdir(os.path.join(input_path, p)))))

    print(f"Total number of sessions: {len(filtered_sessions)}")
    print(filtered_sessions)


    # Model execution
    output_path = "/vast/th3129/data/ibl_new/spectrogram_preprocessed"
    output_path_psd = "/vast/th3129/data/ibl_new/raw_preprocessed"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    for eid in filtered_sessions[start_eid:end_eid]:
        eid_found = any(eid in filename and filename.endswith('_raw.pickle') for filename in os.listdir(output_path))
        if eid_found:
            print(f"skip {eid}")
            continue            

        spectrogram_list = []
        raw_list = []
        psd_list = []
        print(f"Processing Session: {eid}")
        region_counts = [0, 0, 0, 0, 0]

        session_data, sampling_rate, labels = load_session(eid)
        ## calculate the time in seconds in the session_data
        time = len(session_data[1]) / sampling_rate
        print(f"Session {eid} has {time} seconds of data")
        print(session_data[:, :500*2500].shape, session_data.shape)

        savemat(f"/scratch/th3129/region_decoding/data/ibl/raw_signal_mat/raw_{eid}.mat", {"lfp_signal": session_data[:, :500*2500], "labels": labels})

        # preprocess data
        target_sampling_rate = 1250
        preprocessed_data = preprocess_signal(session_data, sampling_rate, target_sampling_rate)
        sampling_rate = target_sampling_rate
        print("sampling rate: ", sampling_rate)
        print(session_data.shape, labels.shape)
        
        for region in hc_acronyms:
            # chan_ids = np.where(labels == region)[0]
            chan_ids = [i for i, label in enumerate(labels) if region in label]
            regional_lfp_data = preprocessed_data[chan_ids]
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
                    # print(f"Shape of row: {row.shape}")
                    z_spect = compute_spectrogram_librosa(row, sampling_rate, num_freq_bins=500, time_bins=16, normalizing='zscore')
                    # db_spect = compute_spectrogram(row, sampling_rate, num_freq_bins=500, time_bins=18, normalizing='db')
                    # print(f"Shape of spectrogram: {z_spect.shape}")
                    
                    y = acr_dict[region]
                    ############################# add count index as trial index
                    if not np.any(np.isnan(z_spect)) and not np.any(np.isinf(z_spect)):
                        spectrogram_list.append((z_spect, y, count, chan_id))  # , lfp_channel_index[i]))
                        raw_list.append((row, y, count, chan_id))  # , lfp_channel_index[i]))
                        count += 1
                        region_counts[acr_dict[region]] += 1

                        base_rms_mini = calculate_rms(row)
                        pow_whole_mini, rms_whole_mini = calculate_power(row, sampling_rate)
                        pow_delta_mini, rms_delta_mini = calculate_power(row, sampling_rate, (0, 4))
                        pow_theta_mini, rms_theta_mini = calculate_power(row, sampling_rate, (4, 8))
                        pow_alpha_mini, rms_alpha_mini = calculate_power(row, sampling_rate, (8, 12))
                        pow_beta_mini, rms_beta_mini = calculate_power(row, sampling_rate, (12, 30))
                        pow_gamma_mini, rms_gamma_mini = calculate_power(row, sampling_rate, (30, -1))
                        probe = 0
                        # Append data to the list
                        psd_list.append(
                            ([
                            base_rms_mini, pow_whole_mini,
                            pow_delta_mini,
                            pow_theta_mini, pow_alpha_mini, pow_beta_mini, pow_gamma_mini, rms_whole_mini,
                            rms_delta_mini,
                            rms_theta_mini, rms_alpha_mini, rms_beta_mini, rms_gamma_mini], acr_dict[region], count, chan_id)
                        )
                    #############################
                    start += window_size
                    end += window_size

        with open(os.path.join(output_path, f"{eid}_data.pickle"), 'wb') as file:
            print(f"Length of data_list: {len(spectrogram_list)}")
            pickle.dump(spectrogram_list, file)
        with open(os.path.join(output_path, f"{eid}_raw.pickle"), 'wb') as file:
            print(f"Length of raw list: {len(raw_list)}")
            pickle.dump(raw_list, file)
        # Convert the list to a DataFrame
        with open(os.path.join(output_path_psd, f"{eid}_data.pickle"), 'wb') as file:
                print(f"Length of data_list: {len(psd_list)}")
                pickle.dump(psd_list, file)
        

        print(region_counts)


# create_spectrogram(chunk_size, start_time, n_chunks)

if __name__ == "__main__":
    args = parse_arguments()
    chunk_size, start_time, n_chunks, start_eid, end_eid = args.chunk_size, args.start_time, args.n_chunks, args.start_eid, args.end_eid
    create_spectrogram(chunk_size, start_time, n_chunks, start_eid, end_eid)