import numba
numba.config.DISABLE_JIT = True
import librosa
import scipy
import torch
import torchaudio
import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset, DataLoader
from scipy.signal import spectrogram
from script.utils import zscore


def compute_trial(raw_signal, sr, ts=2160, t_start=2160, dT=0.5):
    ts_start, ts_end = int((ts - t_start) * sr), int((ts - t_start + dT) * sr)
    return raw_signal[:, ts_start:ts_end]


class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, window_size=1):
        self.spectrograms = spectrograms.astype('float32')
        self.labels = labels
        self.window_size = window_size
        self.n_trials, self.n_channels, self.n_features = spectrograms.shape

    def __len__(self):
        return self.n_trials * self.n_channels

    def __getitem__(self, index):
        trial_idx = index // self.n_channels
        channel_idx = index % self.n_channels

        data = self.spectrograms[trial_idx][channel_idx]
        label = self.labels[channel_idx]

        # select the time neighboring trials of same channel to be positive pairs
        neighbor_trial_indices = list(range(max(0, trial_idx - self.window_size), trial_idx)) + \
                                 list(range(trial_idx + 1, min(self.n_trials, trial_idx + self.window_size + 1)))
        neighbor_trial_idx = random.choice(neighbor_trial_indices)
        augmented_data = self.spectrograms[neighbor_trial_idx][channel_idx]

        return (data, augmented_data), label


class AvgSpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, window_size=1):
        self.spectrograms = spectrograms.astype('float32')
        self.labels = labels
        self.window_size = window_size
        self.n_trials, self.n_channels, self.n_features = spectrograms.shape
        self.avg_spectrograms = np.mean(self.spectrograms, axis=0)

    def __len__(self):
        return self.n_channels

    def __getitem__(self, index):
        data = self.avg_spectrograms[index]
        label = self.labels[index]

        # select the channel of same label to be positive pairs
        same_class_channels_idx = np.arange(self.n_channels)[self.labels == label]
        same_class_channels_idx = np.delete(same_class_channels_idx, np.where(same_class_channels_idx == index))

        if len(same_class_channels_idx) > 0:
            selected_idx = random.choice(same_class_channels_idx)
            augmented_data = self.avg_spectrograms[selected_idx]
        else:
            augmented_data = data

        return (data, augmented_data), label

class OriginalDataset(Dataset):
    def __init__(self, raw_signal, labels, sampling_rate=20000, n_fft=2048, n_freq_bins = 500, n_time_bins=16, transform=None, session_labels=None):
        self.raw_signal = torch.tensor(raw_signal, dtype=torch.float32)
        self.labels = torch.tensor(labels)
        self.n_time_bins = n_time_bins
        self.n_trials, self.n_channels, self.n_features = raw_signal.shape
        self.n_freq_bins, self.n_fft = n_freq_bins, n_fft

        self.hop_length = self.n_features // (self.n_time_bins - 1)
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, power=2.0)
        self.transform = transform
        self.session_labels = session_labels
        self.return_session_trial = True if session_labels is not None else False
        print(self.n_freq_bins, self.n_fft, self.hop_length)

        # precompute spectrograms and augmented spectrograms to save time during training and inference
        self.spectrograms, self.augmented_spectrograms = self._compute_spectrogram(self.raw_signal)
        print("spectrogram of size: ", (self.n_freq_bins, self.n_time_bins))

    def _compute_spectrogram(self, raw_signal):
        spectrograms = torch.zeros((self.n_trials, self.n_channels, self.n_freq_bins*self.n_time_bins))
        augmented_spectrograms = torch.zeros((self.n_trials, self.n_channels, self.n_freq_bins*self.n_time_bins))

        for trial_idx in range(self.n_trials):
            for channel_idx in range(self.n_channels):
                data = raw_signal[trial_idx][channel_idx]
                spectrogram = 10 * torch.log10(self.spectrogram_transform(data)[:self.n_freq_bins])
                spectrograms[trial_idx][channel_idx] = spectrogram.flatten()
                augmented_spectrogram = self.temporal_masking(spectrogram)
                augmented_spectrograms[trial_idx][channel_idx] = augmented_spectrogram.flatten()

        return spectrograms, augmented_spectrograms

    def temporal_masking(self, spectrogram, percent_masked=0.2, mask_size=1):
        n_masks = int(spectrogram.size(1) * percent_masked)

        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(1) - mask_size, (1,)).item()
            spectrogram[:, start_idx:start_idx+mask_size] = 0

        return spectrogram

    def frequency_masking(self, spectrogram, percent_masked=0.05, mask_size=5):
        n_masks = int(spectrogram.size(0) * percent_masked)

        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(0) - mask_size, (1,)).item()
            spectrogram[start_idx:start_idx+mask_size, :] = 0

        return spectrogram


    def __len__(self):
        return self.n_trials * self.n_channels

    def add_gaussian_noise(self, signal):
        return signal + torch.randn_like(signal) * 0.01

    def __getitem__(self, index):
        trial_idx = index // self.n_channels
        channel_idx = index % self.n_channels

        spectrograms = self.spectrograms[trial_idx][channel_idx]
        augmented_spectrograms = self.augmented_spectrograms[trial_idx][channel_idx]

        label = self.labels[channel_idx]

        if self.return_session_trial:
            session_label = self.session_labels[trial_idx]
            return (spectrograms, augmented_spectrograms), label, (session_label, trial_idx)

        return (spectrograms, augmented_spectrograms), label

class RawDataset(Dataset):
    def __init__(self, raw_signal, labels, nperseg=2048, spectrogram_size=500, transform=None, library='pytorch',
                 time_bins=None, vit=False):
        self.raw_signal = torch.tensor(raw_signal, dtype=torch.float32)
        self.labels = torch.tensor(labels)
        self.library = library
        self.sample_rate = 2500  # Make sure to change this according to the sampling rate of the data
        self.n_channels, self.n_features = raw_signal.shape

        # Calculate hop_length based on the desired number of time_bins
        if time_bins is not None:
            total_signal_length = raw_signal.shape[-1]
            if library == 'scipy':
                time_bins = 2 * time_bins + 2
            self.time_bins = time_bins
            hop_length = total_signal_length // (time_bins - 1)
        else:
            hop_length = nperseg // 2

        self.nperseg = nperseg
        self.time_bins = time_bins
        # self.hop_length = hop_length
        self.hop_length = self.n_features // (self.time_bins - 1)
        self.spectrogram_size = spectrogram_size
        self.transform = transform

        # Precompute spectrograms and augmented spectrograms to save time during training and inference
        self.spectrograms, self.augmented_spectrograms = self._compute_spectrogram(self.raw_signal,
                                                                                   spectrogram_size, self.time_bins)
        if vit:
            self.spectrograms = self.vit_preprocess(self.spectrograms)
            self.augmented_spectrograms = self.vit_preprocess(self.augmented_spectrograms)
            spectrogram_size, time_bins=(128, 1024)
        # self.visualize_spectrograms(self.spectrograms, self.augmented_spectrograms, shape=(spectrogram_size, time_bins))
        # self.visualize_mel_spectrograms(self.raw_signal)

    @numba.jit
    def _compute_spectrogram(self, raw_signal, spectrogram_size, timebins):
        spectrograms = torch.zeros((self.n_channels, spectrogram_size * timebins))
        augmented_spectrograms = torch.zeros((self.n_channels, spectrogram_size * timebins))

        for channel_idx in range(self.n_channels):
            data = raw_signal[channel_idx]
            if self.library is None or self.library == 'pytorch':
                spectrogram = self._pytorch_spectrogram(data)
            elif self.library == 'librosa':
                spectrogram = self._librosa_spectrogram(data)
            elif self.library == 'scipy':
                spectrogram = self._scipy_spectrogram(data)
            spectrogram = spectrogram[:spectrogram_size]
            ##### key update that makes the Neuronexus performance improves
            # spectrogram = zscore(spectrogram, axis=0)
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
            spectrograms[channel_idx] = spectrogram.flatten()

            augmented_spectrogram = self.temporal_masking(spectrogram)
            augmented_spectrograms[channel_idx] = augmented_spectrogram.flatten()

        return spectrograms, augmented_spectrograms

    def _pytorch_spectrogram(self, data):
        # Use the torchaudio library for spectrogram computation
        spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.nperseg, win_length=self.nperseg, hop_length=self.hop_length, power=2.0
        )
        return 10 * torch.log10(spectrogram_transform(data))

    # @numba.jit
    def _librosa_spectrogram(self, data):
        # Use librosa for spectrogram computation
        spectrogram = librosa.stft(data.numpy(), n_fft=self.nperseg, hop_length=self.hop_length,
                                   win_length=self.nperseg)
        spectrogram = np.abs(spectrogram) ** 2
        epsilon = 1e-10
        spectrogram = 10 * np.log10(spectrogram + epsilon)
        # spectrogram = zscore(spectrogram, axis=0)
        ## update librosa computation as numpy output
        # spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        return spectrogram

    def _scipy_spectrogram(self, data):
        # Use scipy for spectrogram computation
        frequencies, times, spectrogram = scipy.signal.spectrogram(data.numpy(), nperseg=self.nperseg,
                                                                   noverlap=self.nperseg - self.hop_length)
        return torch.tensor(spectrogram, dtype=torch.float32)

    def vit_preprocess(self, data):
        reshaped_data = data.reshape(self.n_channels, self.spectrogram_size, self.time_bins)
        reshaped_data = torch.tensor(reshaped_data, dtype=torch.float32)

        reshaped_data = F.interpolate(reshaped_data.unsqueeze(1), size=(128, 1024), mode="bilinear", align_corners=False).squeeze(1)
        return reshaped_data

    def temporal_masking(self, spectrogram, percent_masked=0.25, mask_size=1):
        n_masks = int(spectrogram.size(1) * percent_masked)

        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(1), (1,)).item()
            spectrogram[:, start_idx:start_idx + mask_size] = 0

        return spectrogram

    def frequency_masking(self, spectrogram, percent_masked=0.05, mask_size=5):
        n_masks = int(spectrogram.size(0) * percent_masked)

        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(0) - mask_size, (1,)).item()
            spectrogram[start_idx:start_idx + mask_size, :] = 0

        return spectrogram

    def __len__(self):
        return self.n_channels

    def add_gaussian_noise(self, signal):
        return signal + torch.randn_like(signal) * 0.01

    def __getitem__(self, index):
        channel_idx = index % self.n_channels

        spectrograms = self.spectrograms[channel_idx]
        augmented_spectrograms = self.augmented_spectrograms[channel_idx]

        label = self.labels[channel_idx]

        return (spectrograms, augmented_spectrograms), label

    def visualize_spectrograms(self, spectrograms, augmented_spectrograms, shape):
        # Function to save spectrogram
        def save_spectrogram(spectrogram_data, suffix):
            spectrogram = spectrogram_data[0].reshape((shape[0], shape[1]))
            frequencies = np.arange(0, shape[0])

            time_steps = np.linspace(0, 3, shape[1])

            plt.figure(figsize=(6, 6))
            plt.pcolormesh(time_steps, frequencies, spectrogram, shading='gouraud')
            plt.ylim(0, shape[0])
            plt.colorbar(label='Intensity [dB]')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plt.title(f'{suffix} Spectrogram - Normal Scale')
            plt.tight_layout()
            plt.savefig(f'../results/{suffix}_spectrogram_normal.png')

        # Save normal spectrogram
        save_spectrogram(spectrograms, 'normal')

        # Save Mel spectrogram
        save_spectrogram(augmented_spectrograms, 'augmented')

    def visualize_mel_spectrograms(self, raw_signal):
        def save_mel_spectrogram(raw_signal, suffix, sampling_rate=None):
            raw_signal = raw_signal[0].numpy()
            if sampling_rate is None:
                sampling_rate = self.sample_rate

            # Generate Mel spectrogram using librosa
            mel_spec = librosa.feature.melspectrogram(y=raw_signal, sr=sampling_rate, n_fft=self.nperseg,
                                                      hop_length=self.hop_length, n_mels=128, fmax=500, fmin=0)

            # Convert to decibels (dB) for better visualization
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Time steps and frequency bins for the Mel spectrogram
            time_steps = np.linspace(0, 3, 16)
            frequencies = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=500)

            plt.figure(figsize=(6, 6))
            plt.pcolormesh(time_steps, frequencies, mel_spec_db, shading='gouraud')
            plt.colorbar(label='Intensity [dB]')
            # every tenth frequency
            plt.yticks(frequencies[::10], frequencies[::10].astype(int))
            plt.title(f'{suffix} Mel Spectrogram')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plt.tight_layout()
            plt.savefig(f'../results/{suffix}_mel_spectrogram.png')

        # Save Mel spectrogram
        save_mel_spectrogram(raw_signal, 'normal')


class ImageDataset(Dataset):
    """
    Similar as RawDataset, except for outputing 2d images instead of 1d flattened images
    """

    def __init__(self, raw_signal, labels, sampling_rate=20000, n_fft=2048, n_freq_bins=500, n_time_bins=16,
                 transform=None, session_labels=None, augmentation_type=None, aug_params=None):
        self.raw_signal = torch.tensor(raw_signal, dtype=torch.float32)
        self.labels = torch.tensor(labels)
        self.n_time_bins = n_time_bins
        self.n_trials, self.n_channels, self.n_features = raw_signal.shape
        self.n_freq_bins, self.n_fft = n_freq_bins, n_fft

        self.hop_length = self.n_features // (self.n_time_bins - 1)
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.n_fft,
                                                                       hop_length=self.hop_length, power=2.0)
        self.transform = transform
        self.session_labels = session_labels
        self.return_session_trial = True if session_labels is not None else False

        # precompute spectrograms and augmented spectrograms to save time during training and inference
        self.spectrograms = self._compute_spectrogram(self.raw_signal)
        self.augmentation = Augmentation(augmentation_type, **(aug_params or {}))
        print("spectrogram of size: ", (self.n_freq_bins, self.n_time_bins))

    def _compute_spectrogram(self, raw_signal):
        spectrograms = torch.zeros((self.n_trials, self.n_channels, self.n_freq_bins, self.n_time_bins))

        for trial_idx in range(self.n_trials):
            for channel_idx in range(self.n_channels):
                data = raw_signal[trial_idx][channel_idx]
                spectrogram = 10 * torch.log10(self.spectrogram_transform(data)[:self.n_freq_bins])
                spectrograms[trial_idx][channel_idx] = spectrogram

        return spectrograms

    def __len__(self):
        return self.n_trials * self.n_channels

    def get_augmentation(self, trial_idx, channel_idx):
        return self.augmentation.apply(
            self.spectrograms[trial_idx][channel_idx],
            trial_idx=trial_idx,
            channel_idx=channel_idx,
            dataset=self
        )

    def __getitem__(self, index):
        trial_idx = index // self.n_channels
        channel_idx = index % self.n_channels

        spectrograms = self.spectrograms[trial_idx][channel_idx].unsqueeze(0)
        augmented_spectrograms = self.get_augmentation(trial_idx, channel_idx).unsqueeze(0)
        label = self.labels[channel_idx]

        if self.return_session_trial:
            session_label = self.session_labels[trial_idx]
            return (spectrograms, augmented_spectrograms), label, (session_label, trial_idx)

        return (spectrograms, augmented_spectrograms), label

    # def get_augmentation(self, trial_idx, channel_idx):
    #     if self.temporal_neighbors:
    #         neighbor_trial_idx = (trial_idx + 1) % self.n_trials
    #         neighbor_channel_idx = (channel_idx + 1) % self.n_channels
    #         if random.random() < self.augmentation_prob:
    #             augmented_spectrograms = self.spectrograms[neighbor_trial_idx][channel_idx].unsqueeze(0)
    #         else:
    #             augmented_spectrograms = self.spectrograms[trial_idx][neighbor_channel_idx].unsqueeze(0)
    #     else:
    #         augmented_spectrograms = self.augmented_spectrograms[trial_idx][channel_idx].unsqueeze(0)
    #     return augmented_spectrograms


class Augmentation:
    def __init__(self, augmentation_type=None, **kwargs):
        self.augmentation_type = augmentation_type
        self.params = kwargs

    def apply(self, spectrogram, trial_idx=None, channel_idx=None, dataset=None):
        if self.augmentation_type == "temporal_masking":
            return self.temporal_masking(spectrogram, **self.params)
        elif self.augmentation_type == "frequency_masking":
            return self.frequency_masking(spectrogram, **self.params)
        elif self.augmentation_type == "temporal_neighbors":
            return self.temporal_neighbors(trial_idx, channel_idx, dataset)
        elif self.augmentation_type == "spatial_neighbors":
            return self.spatial_neighbors(trial_idx, channel_idx, dataset)
        else:
            return spectrogram

    @staticmethod
    def temporal_masking(spectrogram, percent_masked=0.2, mask_size=1):
        n_masks = int(spectrogram.size(1) * percent_masked)
        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(1) - mask_size, (1,)).item()
            spectrogram[:, start_idx:start_idx + mask_size] = 0
        return spectrogram

    @staticmethod
    def frequency_masking(spectrogram, percent_masked=0.05, mask_size=5):
        n_masks = int(spectrogram.size(0) * percent_masked)
        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(0) - mask_size, (1,)).item()
            spectrogram[start_idx:start_idx + mask_size, :] = 0
        return spectrogram

    def temporal_neighbors(self, trial_idx, channel_idx, dataset):
        neighbor_trial_idx = (trial_idx + 1) % len(dataset.spectrograms)
        return dataset.spectrograms[neighbor_trial_idx][channel_idx]

    def spatial_neighbors(self, trial_idx, channel_idx, dataset):
        neighbor_channel_idx = (channel_idx + 1) % dataset.n_channels
        return dataset.spectrograms[trial_idx][neighbor_channel_idx]
