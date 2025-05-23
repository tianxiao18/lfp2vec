import h5py
from lxml import etree
import spikeinterface as si
import json
import numpy as np
import pandas as pd
from spikeinterface import NumpyRecording
from spikeinterface.preprocessing import bandpass_filter
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import os


def load_large_signal(raw_signal_path, sample_rate=20000, T=30, downsample_factor=4):
    with h5py.File(raw_signal_path, 'r') as file:
        raw_signal = file['data']

        if downsample_factor: # load and downsample in chunk
                processed_path = os.path.splitext(raw_signal_path)[0] + '.npy'
                if os.path.isfile(processed_path):
                    resampled_signal = np.load(processed_path)
                else:
                    n_channel, n_timepoints = raw_signal.shape
                    n_timepoints_resampled = n_timepoints // downsample_factor

                    resampled_signal = np.zeros((len(raw_signal), n_timepoints_resampled))
                    for start in tqdm(range(0, n_timepoints, sample_rate)):
                        # here we use sample_rate as batch_size for loading
                        end = min(start+sample_rate, n_timepoints)
                        chunk = raw_signal[:, start:end]

                        for i in range(n_channel):
                            resampled_chunk = downsample(chunk[i], sample_rate, downsample_factor)
                            resampled_signal[i, start//downsample_factor:end//downsample_factor] = resampled_chunk
        
                    np.save(processed_path, resampled_signal)
                return resampled_signal
        
        # if not downsample, load first T seconds of raw_signal 
        return raw_signal[:, :int(sample_rate*T)]
    

def downsample(raw_signal, fs_source, downsampling_factor):
    # apply low-pass butterworth filter
    fs_target = fs_target / downsampling_factor
    nyquist_target = fs_target / 2
    b, a = butter(5, nyquist_target / (fs_source / 2), btype='low')
    filtered_signal = filtfilt(b, a, raw_signal)

    # downsample by approximate factor
    downsampled_signal = filtered_signal[::downsampling_factor]
    return downsampled_signal


def load_data(raw_signal_path, label_path, xml_path, sheet_name, sample_rate=20000, T=30):

    # load raw data from mat file
    raw_signal = load_large_signal(raw_signal_path, sample_rate, T, downsample_factor=None)
    
    # load region labels from excel
    df = pd.read_excel(label_path, sheet_name=sheet_name)

    # load xml file to find skipped channels
    tree = etree.parse(xml_path)
    root = tree.getroot()
    element = root.find('anatomicalDescription')

    # find all channels to be skipped (1 = skipped, 0 = not skip)
    skip_arr = np.zeros((1024, ))

    for channel in element.findall('.//channel'):
        skip_value = channel.attrib['skip']
        c = int(channel.text)
        skip_arr[c] = skip_value

    skipped_channels = np.where(skip_arr == 1)[0]

    return raw_signal, df, skipped_channels


def process_labels(df, mapping_path, skipped_channels):
    """
    Here we obtain a channel-region map with bad channels skipped
    """
    with open(mapping_path) as json_file:
        mapping = json.load(json_file)
    
    channels = pd.concat([df.iloc[:, i] for i in range(0, len(df.columns), 2)], ignore_index=True)
    regions = pd.concat([df.iloc[:, i+1] for i in range(0, len(df.columns), 2)], ignore_index=True)

    output = pd.DataFrame({"channels": channels, "regions":regions})
    output["channels"] -= 1

    # fill in NaN channels with UNK
    for idx, row in output.iterrows():
        if pd.isna(row["channels"]):
            output.at[idx, "channels"] = output.at[idx-1, "channels"]-1
        if pd.isna(row["regions"]):
            output.at[idx, "regions"] = "UNK"

    # remove bad channels in skipped_channels
    processed_channels = np.array(output["channels"]).astype(int)
    mask = ~output["channels"].isin(skipped_channels)
    output = output[mask]

    # fix diverse labels
    output = output.replace(mapping)

    # fix shuffled channel number
    corrected_channels = np.array([list(range(shank*128+127, shank*128-1, -1)) for shank in range(8)]).flatten()
    channel_channel_map = dict(zip(processed_channels, corrected_channels))
    output = output.replace(channel_channel_map)

    output["channels"] = output["channels"].astype(int)

    # update skipped_channels based on shuffled channel number
    vectorized_func = np.vectorize(channel_channel_map.get)
    skipped_channels = vectorized_func(skipped_channels) if len(skipped_channels) > 0 else skipped_channels

    return output, skipped_channels, channel_channel_map


def process_signals(raw_signal, channel_channel_map, normalize=True):
    # n_channel * n_samples (30 s*20000 Hz)
    raw_signal = raw_signal.astype('float32')
    processed_signal = np.zeros_like(raw_signal)

    for i in tqdm(range(len(raw_signal))):
        j = channel_channel_map[i]
        if normalize: 
            if np.std(raw_signal[i]) != 0:
                processed_signal[j] = (raw_signal[i] - np.mean(raw_signal[i])) / np.std(raw_signal[i])
        else:
            processed_signal[j] = raw_signal[i]

    return processed_signal


def filter_signals(raw_signal, freq_min=60, freq_max=200):
    recording = NumpyRecording(raw_signal.T, sampling_frequency=20000)
    filtered_recording = bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
    filtered_signals = filtered_recording.get_traces(0, 0, raw_signal.shape[1]).T
    return filtered_signals