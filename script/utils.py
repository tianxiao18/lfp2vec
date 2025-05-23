import os
import librosa
import pickle
import numpy as np

from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from collections import Counter
from scipy.signal import welch
from scipy.signal import spectrogram
import torch

def calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
                                test_pred_labels_all, sessions_list, title):
    # Train metrics
    for i in range(len(train_real_labels_all)):
        train_real_labels = train_real_labels_all[i]
        train_pred_labels = train_pred_labels_all[i]

        bal_train_acc = balanced_accuracy_score(train_real_labels, train_pred_labels)
        train_acc = accuracy_score(train_real_labels, train_pred_labels)
        bal_train_f1_score = f1_score(train_real_labels, train_pred_labels, average='macro')
        train_f1_score = f1_score(train_real_labels, train_pred_labels, average='micro')

        label_counts = Counter(train_real_labels)
        most_frequent_label_count = label_counts.most_common(1)[0][1]
        chance_level_accuracy = most_frequent_label_count / len(train_real_labels)

        print(f"Title: {title}, Train Session: {sessions_list[i]}, CA: {chance_level_accuracy:.5f}, "
              f"BTA: {bal_train_acc:.5f}, BTF1: {bal_train_f1_score:.5f}, "
              f"TA: {train_acc:.5f}, TF1: {train_f1_score:.5f}")

    # Test metrics
    for i in range(len(test_real_labels_all)):
        test_real_labels = test_real_labels_all[i]
        test_pred_labels = test_pred_labels_all[i]

        bal_test_acc = balanced_accuracy_score(test_real_labels, test_pred_labels)
        test_acc = accuracy_score(test_real_labels, test_pred_labels)
        bal_test_f1_score = f1_score(test_real_labels, test_pred_labels, average='macro')
        test_f1_score = f1_score(test_real_labels, test_pred_labels, average='micro')

        label_counts = Counter(test_real_labels)
        most_frequent_label_count = label_counts.most_common(1)[0][1]
        chance_level_accuracy = most_frequent_label_count / len(test_real_labels)

        print(f"Title: {title}, Test Session: {sessions_list[i]}, CA: {chance_level_accuracy:.5f}, "
              f"BTA: {bal_test_acc:.5f}, BTF1: {bal_test_f1_score:.5f}, "
              f"TA: {test_acc:.5f}, TF1: {test_f1_score:.5f}")



def zscore(a, axis):
    """Compute the z-score normalization along the specified axis."""
    if isinstance(a, np.ndarray):
        mean = a.mean(axis=axis, keepdims=True)
        std = a.std(axis=axis, ddof=0, keepdims=True)
    elif isinstance(a, torch.Tensor):
        mean = a.mean(dim=axis, keepdim=True)
        std = a.std(dim=axis, unbiased=False, keepdim=True)

    std[std == 0] = 1.0  # Prevent division by zero
    z = (a - mean) / std
    return z


def compute_spectrogram(x, fs, num_freq_bins=500, time_bins=18, normalizing='db', return_onesided=True, **kwargs):
    nperseg = 2 * (num_freq_bins - 1)
    noverlap = int(nperseg - len(x) / time_bins)

    f, t, Zxx = spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap, return_onesided=return_onesided, **kwargs)

    Zxx = Zxx[:num_freq_bins, :]
    f = f[:num_freq_bins]

    # Take the magnitude of the complex values
    Zxx = np.abs(Zxx)

    # Apply normalization
    if normalizing == "zscore":
        Zxx = zscore(Zxx, axis=-1)
        if (Zxx.std() == 0).any():
            Zxx = np.ones_like(Zxx)
        # Zxx = Zxx[:, 10:-10]  # Optional cropping of frequencies
    elif normalizing == "db":
        Zxx = np.log(Zxx + 1e-10)  # Log scale to avoid log(0)

    # Replace NaNs with zero
    if np.isnan(Zxx).any():
        Zxx = np.nan_to_num(Zxx, nan=0.0)

    # plt.figure(figsize=(10, 6))
    # plt.pcolormesh(t, f, Zxx, shading='gouraud', cmap='inferno')
    # plt.colorbar(label='Power (dB)')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (s)')
    # plt.title('Spectrogram')
    # plt.savefig("spectrogram.png", dpi=300, bbox_inches='tight')
    # plt.close()

    return np.transpose(Zxx)


def save_if_better(output_path, val_acc, train_real_labels, train_pred_labels, test_real_labels, test_pred_labels):
    # Check if the file exists
    if os.path.exists(output_path):
        # Load the existing pickle file
        with open(output_path, 'rb') as f:
            saved_data = pickle.load(f)
            saved_val_acc = saved_data['decoder_val_acc']

        # Compare the current test_acc with the saved one
        if val_acc > saved_val_acc:
            print(f"New test accuracy ({val_acc}) is higher than saved accuracy ({saved_val_acc})"
                  f", overwriting the file.")
            with open(output_path, 'wb') as f:
                pickle.dump({'decoder_val_acc': val_acc, 'train_real_labels': train_real_labels,
                             'train_pred_labels': train_pred_labels, 'test_real_labels': test_real_labels,
                             'test_pred_labels': test_pred_labels}, f)
        else:
            print(f"Saved test accuracy ({saved_val_acc}) is higher, keeping the old file.")
    else:
        # If the file doesn't exist, create it
        with open(output_path, 'wb') as f:
            print(f"No existing file found. Saving new file with test accuracy: {val_acc}.")
            pickle.dump({'decoder_val_acc': val_acc, 'train_real_labels': train_real_labels,
                         'train_pred_labels': train_pred_labels, 'test_real_labels': test_real_labels,
                         'test_pred_labels': test_pred_labels}, f)

def compute_spectrogram_librosa(x, fs, num_freq_bins=500, time_bins=16, axis=0, **kwargs):
    nperseg = 2048
    total_signal_length = x.shape[0]
    hop_length = total_signal_length // (time_bins - 1)
    spectrogram = librosa.stft(x, n_fft=nperseg, hop_length=hop_length,
                                   win_length=nperseg)
    spectrogram = np.abs(spectrogram) ** 2
    epsilon = 1e-10
    spectrogram = 10 * np.log10(spectrogram + epsilon)
    spectrogram = spectrogram[:500]

    return spectrogram



def calculate_psd(data, sampling_rate, freq_range=None):
    freqs, Pxx = welch(data, fs=sampling_rate, nperseg=1024)

    if freq_range is not None:
        min_freq, max_freq = freq_range
        if max_freq == -1:
            mask = (freqs >= min_freq)
        else:
            mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs = freqs[mask]
        Pxx = Pxx[mask]

    return freqs, Pxx


def calculate_power(data, sampling_rate, freq_range=None):
    freqs, Pxx = calculate_psd(data, sampling_rate, freq_range)

    power = np.trapz(Pxx, freqs)
    rms = calculate_rms(Pxx)
    return power, rms


def calculate_rms(data):
    return np.sqrt(np.mean(data ** 2))
