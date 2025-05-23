import sys
sys.path.append("/scratch/th3129/region_decoding")
import scipy
import scipy.io
import pickle

from script.utils import *
from blind_localization.data.data_loading import *
from blind_localization.data.preprocess import *
from script.ibl_preprocessing.preprocess_psd_spectrogram import preprocess_signal

def load_signal_labels(source_session, sample_rate=20000, T=30, data_type='lfp'):
    source_file = load_session_data("/scratch/th3129/region_decoding/script/Neuronexus_preprocessing/file_path_hpc.json", source_session)
    public_file = load_session_data("/scratch/th3129/region_decoding/script/Neuronexus_preprocessing/file_path_hpc.json", "public")
    signal_path = source_file["raw_lfp_path"] if data_type == 'lfp' else source_file["raw_signal_path"]
    print(signal_path)
    
    raw_signal, df, skipped_channels = load_data(signal_path, public_file["label_path"], source_file["xml_path"], sheet_name=source_file["sheet_name"],
                                                 sample_rate=sample_rate, T=T)
    channel_region_map, skipped_channels, channel_channel_map = process_labels(df, public_file["mapping_path"], skipped_channels)
    raw_signal = process_signals(raw_signal, channel_channel_map, normalize=False)
    raw_signal = preprocess_signal(raw_signal, original_sampling_rate=sample_rate, target_sampling_rate=sample_rate, low_pass=True)
    raw_signal = raw_signal * 0.195 / 1000 / 1000     # convert ADC units to volts

    channel_labels = np.load(f"/scratch/cl7201/region_decoding/data/Neuronexus/labels_{source_session}.npy")
    labels = np.argmax(channel_labels, axis=1) + 1
    channel_labels = channel_labels * labels[:, np.newaxis]
    class_labels = np.sum(channel_labels, axis=1).astype(int)
    
    dict = {0:5, 1:0, 2:1, 3:2, 4:3, 5:4}
    class_labels = np.array([dict[c] for c in class_labels])

    label_dict = {0:4, 1:0, 2:1, 3:2, 4:3, 5:5}
    class_labels = np.array([label_dict[c] for c in class_labels])

    swr_file = scipy.io.loadmat(source_file["swr_path"])
    swr_timestamp = swr_file['ripples']['timestamps'][0][0]
  
    return raw_signal, class_labels, swr_timestamp


def preprocess_data(sessions, session_config):
    t_start = session_config['t_starts'][0]
    dT = session_config['trial_length']
    region_counts = [0, 0, 0, 0, 0]
    for source_session in sessions:
        T = session_config['t_starts'][-1] - session_config['t_starts'][0] + (session_config['t_starts'][1] -  session_config['t_starts'][0])
        print(f"Session length: {T} sec")
        n_time_bins = 16

        file_path_spectrogram = os.path.join(session_config['folder_path'], 'spectrogram')
        file_path_raw = os.path.join(session_config['folder_path'], 'raw')
        file_path_lfp = os.path.join(session_config['folder_path'], 'lfp')

        # store X of shape (n_trials, n_channels, n_features) and y of shape (n_channels, )
        
        lfp_signal, channel_labels, swr_timestamp = load_signal_labels(source_session, sample_rate=1250, T=T, data_type='lfp')
        raw_signal, channel_labels, swr_timestamp = load_signal_labels(source_session, sample_rate=20000, T=T, data_type='raw')
        print(lfp_signal.shape, raw_signal.shape)

        data_list = []
        raw_list = []
        feature_list = []
        lfp_list = []
        # iterate over channels
        for i, (channel_data, label) in tqdm(enumerate(zip(raw_signal, channel_labels))):
            
            # if i==5: skip
            if label == 5:
                continue
            lfp_channel_data = lfp_signal[i]

            for count, ts in enumerate(session_config['t_starts']):
                ts_start, ts_end = int((ts-t_start)*20000), int((ts-t_start+dT)*20000)
                row = channel_data[ts_start:ts_end]
                
                ts_start_lfp, ts_end_lfp = int((ts-t_start)*1250), int((ts-t_start+dT)*1250)
                row_lfp = lfp_channel_data[ts_start_lfp:ts_end_lfp]

                z_spect = compute_spectrogram_librosa(row_lfp, 1250, num_freq_bins=500, time_bins=16, normalizing='zscore')
                if not np.any(np.isnan(z_spect)) and not np.any(np.isinf(z_spect)):
                    data_list.append((z_spect, label, count, i))  # , lfp_channel_index[i]))
                    raw_list.append((row, label, count, i))  # , lfp_channel_index[i]))
                    lfp_list.append((row_lfp, label, count, i))  # , lfp_channel_index[i]))                    
                    region_counts[label] += 1
                

                ##### raw
                if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                    base_rms_mini = calculate_rms(row)
                    pow_whole_mini, rms_whole_mini = calculate_power(row, 20000)
                    pow_delta_mini, rms_delta_mini = calculate_power(row, 20000, (0, 4))
                    pow_theta_mini, rms_theta_mini = calculate_power(row, 20000, (4, 8))
                    pow_alpha_mini, rms_alpha_mini = calculate_power(row, 20000, (8, 12))
                    pow_beta_mini, rms_beta_mini = calculate_power(row, 20000, (12, 30))
                    pow_gamma_mini, rms_gamma_mini = calculate_power(row, 20000, (30, -1))
    
                    # Append data to the list
                    feature_list.append(
                        ([
                        base_rms_mini, pow_whole_mini,
                        pow_delta_mini,
                        pow_theta_mini, pow_alpha_mini, pow_beta_mini, pow_gamma_mini, rms_whole_mini,
                        rms_delta_mini,
                        rms_theta_mini, rms_alpha_mini, rms_beta_mini, rms_gamma_mini], label, count, i)
                    )

        # spectrogram 
        with open(os.path.join(file_path_spectrogram, f"{source_session}_data.pickle"), 'wb') as file:
            print(f"Length of data list: {len(data_list)}")
            pickle.dump(data_list, file)

        # raw signal (with both LFP and AP)
        with open(os.path.join(file_path_lfp, f"{source_session}_raw.pickle"), 'wb') as file:
            print(f"Length of raw list: {len(raw_list)}")
            pickle.dump(raw_list, file)

        # LFP features (alpha, beta, theta, gamma power)
        with open(os.path.join(file_path_raw, f"{source_session}_data.pickle"), 'wb') as file:
            print(f"Length of feature list: {len(feature_list)}")
            pickle.dump(feature_list, file)

        # LFP signal only
        with open(os.path.join(file_path_lfp, f"{source_session}_lfp.pickle"), 'wb') as file:
            print(f"Length of lfp list: {len(lfp_list)}")
            pickle.dump(lfp_list, file)
        

def main():
    folder_path = "/scratch/th3129/region_decoding/data/Neuronexus"
    session_config = {
        "visualize": True,
        'swr_flag': False,
        't_starts': np.arange(2160, 2340, 3),
        'trial_length': 3,
        'folder_path': folder_path
    }
    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    preprocess_data(session_names, session_config)

if __name__ == "__main__":
    main()