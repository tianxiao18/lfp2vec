import torch
import torchaudio
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class GraphDataset(InMemoryDataset):
    """
    define a Graph dataset, here each data point represents one graph G = (X, A) for one trial,
    where X = spectrogram features of each channel, A = functional correlation between each channels
    """
    def __init__(self, raw_signal, adj_matrix, labels, sampling_rate=20000, n_fft=2048, n_freq_bins = 500, global_positions=None, channel_indices=None):
        super(GraphDataset, self).__init__(None, transform=None, pre_transform=None)

        self.raw_signal = torch.tensor(raw_signal, dtype=torch.float32)
        self.labels = torch.tensor(labels)
        self.adj_matrix = [torch.tensor(mtx) for mtx in adj_matrix]
        
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=n_fft, hop_length=n_fft//2, power=2.0)
        self.n_trials, self.n_channels, self.n_features = raw_signal.shape
        self.n_freq_bins, self.n_fft = n_freq_bins, n_fft
        self.global_positions = global_positions
        self.channel_indices = channel_indices

        # precompute spectrograms to save time during training and inference
        self.spectrograms = self._compute_spectrogram(self.raw_signal)
        self.visualize_spectrograms(self.spectrograms, sampling_rate, name='graph')

    def _compute_spectrogram(self, raw_signal):
        """
        Given raw signal of shape (n_trials, n_channels, n_timebins), 
        return spectrogram of shape (n_trials, n_channels, n_features) 
        """
        self.n_time_bins = self.spectrogram_transform(raw_signal[0][0]).size(1)
        spectrograms = torch.zeros((self.n_trials, self.n_channels, self.n_freq_bins*self.n_time_bins))

        for trial_idx in range(self.n_trials):
            for channel_idx in range(self.n_channels):
                data = raw_signal[trial_idx][channel_idx]
                spectrogram = 10 * torch.log10(self.spectrogram_transform(data)[:self.n_freq_bins])
                spectrograms[trial_idx][channel_idx] = spectrogram.flatten()
                
        return spectrograms
    
    def len(self):
        return self.n_trials
    
    def get(self, trial_idx):
        "retrive an entire graph G = (X, A), Y representing one trial"
        spectrograms = self.spectrograms[trial_idx]
        edges = self.adj_matrix[trial_idx]
        labels = self.labels
        pos = self.global_positions[self.channel_indices[trial_idx]] if self.channel_indices is not None else None

        data=Data(x=spectrograms, edge_index=edges, y=labels, pos=pos)
        return data

    def visualize_spectrograms(self, spectrograms, sample_rate, name):
        spectrogram = spectrograms[0][0].reshape((self.n_freq_bins, self.n_time_bins))

        frequencies = np.linspace(0, sample_rate/2, self.n_fft//2+1)[:self.n_freq_bins]
        time_steps = np.linspace(0, self.n_features/sample_rate, self.n_time_bins)

        # Plot using pcolormesh
        font = {'size': 18}
        matplotlib.rc('font', **font)

        plt.figure()
        plt.pcolormesh(time_steps, frequencies, spectrogram, shading='gouraud')
        plt.ylim(0, 500)
        plt.colorbar(label='Intensity [dB]')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.savefig(f'results/{name}_spectrogram.png')

def zscore(a, axis):
    """Compute the z-score normalization along the specified axis."""
    mn = a.mean(dim=axis, keepdims=True)
    std = a.std(dim=axis, unbiased=False, keepdim=True)

    std[std == 0] = 1.0  # Prevent division by zero
    z = (a - mn) / std
    return z