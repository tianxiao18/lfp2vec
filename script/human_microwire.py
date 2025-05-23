import h5py
import math
import numpy as np
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import spectrogram
import pandas as pd
from sklearn.decomposition import PCA

def load_microwire_recordings(dir_path):
    raw_signals = []
    channel_names = []

    for file in os.listdir(dir_path):
        # only focus on micro wires, not micro contact such as LFA-LFC
        if 'Micro' in file:
            file_path = os.path.join(dir_path, file)

            with h5py.File(file_path, 'r') as file:
                Fs = math.floor(1 / file[next(iter(file))]['interval'][0][0])
                n_samples = file[next(iter(file))]['length'][0][0]
                n_channels = len(file.keys())
                print(f'{n_channels} channel signal of {n_samples/Fs/60/60:.2f}h sampled at {Fs}Hz ')

                channel_name = []
                raw_signal = np.zeros((n_channels, Fs*300))

                for i, channel in enumerate(file.keys()):
                    print(channel)
                    channel_name.append(channel)
                    data = file[channel]['values'][0, :Fs*300]
                    raw_signal[i] = (data - np.mean(data)) / np.std(data)

                raw_signals.append(raw_signal)
                channel_names.extend(channel_name)

    raw_signals = np.vstack(raw_signals)
    return raw_signals, channel_names

def load_labels(label_path):
    df = pd.read_csv(label_path, sep=' ', header=None, names=['name','x','y','z','region'],index_col=False)
    return df

def compute_spectrogram_raw(raw_signal, sr, nperseg=2048, dT=0.5):
    spectrograms = np.zeros((len(raw_signal), 1025, 8))
    ts_start, ts_end = 0, int(dT*sr)

    for channel in np.arange(len(raw_signal)):
        f, t, Sxx = spectrogram(raw_signal[channel, ts_start:ts_end], sr, nperseg=nperseg)
        spectrograms[channel] = 10 * np.log10(Sxx)

    return spectrograms, f, t

def visualize_spectrograms(spectrograms, f, t):
    plt.figure(figsize=(40, 32))
    cnt = 1
    for channel in np.arange(len(spectrograms)):
        plt.subplot(4, 4, cnt)
        plt.pcolormesh(t, f, spectrograms[channel], shading='gouraud', vmin=-100, vmax=-5)
        plt.colorbar(label='Intensity [dB]')
        plt.ylim(0, 500)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title(f'Spectrogram of channel {channel}')
        cnt += 1
    plt.tight_layout()
    plt.savefig('results/spectrogram.png')

def visualize_raw_signals(raw_signal, channel_names, T = 1000, Fs = 4095):
    offset = 5
    sr_converted = int(Fs/1000) #samples/ms
    n_channels = len(channel_names)

    time = np.arange(0, T, 1/sr_converted)
    channel_names = [channel_names[i].split('_')[-1] for i in range(len(channel_names))]
    plt.figure(figsize=(32, 16))
    for i in range(n_channels): 
        plt.plot(time, raw_signal[i, :T*sr_converted]+offset*i, c='black')
    
    plt.xlabel("time(ms)")
    plt.ylabel("channel")
    plt.yticks(offset*np.arange(n_channels), channel_names)
    plt.savefig('results/microwire_signal.png')

def visualize_channel_locations(label_df_ls, names):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, label_df in enumerate(label_df_ls):
        # label_df = label_df[label_df['name'].str.contains('MICRO')].reset_index(drop=True)
        ax.scatter(label_df['x'], label_df['y'], label_df['z'], s=50, alpha=0.8, edgecolor='k', label=names[i])
        for j in range(len(label_df)):
            ax.text(label_df['x'][j], label_df['y'][j], label_df['z'][j]+0.5, label_df['name'][j], size=7, zorder=1, color='k')
    
    plt.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('results/channel_locations.png', dpi=200)

    combined_df = pd.concat(label_df_ls)
    value_counts = combined_df['region'].value_counts()
 
    plt.figure(figsize=(10, 8))
    ax = value_counts.plot(kind='bar', color='skyblue', alpha=0.7)
    ax.set_xticklabels(value_counts.index, rotation=0, ha='center')

    for i in range(len(value_counts)):
        ax.text(i, value_counts.iloc[i] + 0.1, str(value_counts.iloc[i]), ha='center')

    plt.xlabel('region')
    plt.ylabel('n_channel')
    plt.title('Histogram of channel regions in micro-wire recordings')
    plt.savefig('results/labels_hist.png')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(value_counts)):
        region_df = combined_df[combined_df['region'] == value_counts.index[i]]
        ax.scatter(region_df['x'], region_df['y'], region_df['z'], s=50, alpha=0.8, edgecolor='k', label=value_counts.index[i])
            
    plt.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('results/channel_region_locations.png', dpi=200)

def visualize_all_channels(dir_path):
    labels_path_ls = []
    label_df_ls = []
    names = []

    for folder in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if '.txt' in file:
                    file_path = os.path.join(folder_path, file)
                    labels_path_ls.append(file_path)
                    names.append(file[:-4])

    for labels_path in labels_path_ls:
        label_df = load_labels(labels_path)
        label_df_ls.append(label_df)

    visualize_channel_locations(label_df_ls, names)

def main():
    dir_path = "/scratch/th3129/shared/human_microwire"
    file_path = "/scratch/th3129/shared/human_microwire/NY6_matlab"
    raw_signals, channel_names = load_microwire_recordings(file_path)
    print(channel_names)
    # visualize_raw_signals(raw_signals, channel_names, T = 1000, Fs = 32767)

    label_path = "/scratch/th3129/shared/human_microwire/NY6_preOP/NYR6_T1_depth_ASHS_HPC_subfields.txt"
    label_df = load_labels(label_path)
    # visualize_all_channels(dir_path)

    spectrograms, f, t = compute_spectrogram_raw(raw_signals, sr=32767, nperseg=2048)
    print(spectrograms.shape)
    # visualize_spectrograms(spectrograms, f, t)

    X = spectrograms.reshape(spectrograms.shape[0], -1)
    pca = PCA(n_components=3)
    x_hat = pca.fit_transform(X)

    ax = plt.subplot(111, projection='3d')
    ax.scatter(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], s=30, c=channel_names)
    plt.legend()
    plt.savefig('pca.png')

if __name__ == "__main__":
    main()
