import pickle
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class LFP2VecDataset(Dataset):
    """Tensor-friendly dataset for LFP-to-vec experiments.

    Stores per-sample waveform, label, and a trial-channel identifier string.
    Optionally supports upsampling raw signals to the target sampling rate.
    """
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        chans: list[str],
    ) -> None:
        self.data = data
        self.labels = labels
        self.chans = chans

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        return self.data[index], self.labels[index]

    def get_trial_chan(self, index: int) -> str:
        return self.chans[index]

    def upsample_data(self, signal_sampling_rate: int) -> None:
        """Resample each 1D signal to 16 kHz and z-score normalize per sample."""
        target_sampling_rate = 16000
        epsilon = 1e-10
        upsampled_signals = []

        for signal in self.data:
            num_target_samples = int(
                len(signal) * target_sampling_rate / signal_sampling_rate
            )
            upsampled_signal = resample(signal, num_target_samples)
            upsampled_signal = (upsampled_signal - np.mean(upsampled_signal)) / (
                np.std(upsampled_signal) + epsilon
            )
            upsampled_signals.append(upsampled_signal)

        self.data = torch.as_tensor(np.array(upsampled_signals), dtype=torch.float32)


class LFP2VecDataLoader:
    """Utility to load per-session pickled data and build split datasets.

    It loads samples per session, removes all-zero entries, concatenates
    across sessions, and performs train/val/test splits while keeping
    trial-channel identifiers aligned with labels.
    """

    DATA_SUITS = {
        "Allen": {
            "sessions_list": [
                "719161530",
                "794812542",
                "778998620",
                "798911424",
                "771990200",
                "771160300",
                "768515987",
            ],
            "pickle_path": "spectrogram/Allen",
            "hc_acronyms": ["CA1", "CA2", "CA3", "DG", "VIS"],
        },
        "ibl": {
            "sessions_list": [
                "0802ced5-33a3-405e-8336-b65ebc5cb07c_probe00",
                "0802ced5-33a3-405e-8336-b65ebc5cb07c_probe01",
                "0a018f12-ee06-4b11-97aa-bbbff5448e9f_probe00",
                "3638d102-e8b6-4230-8742-e548cd87a949_probe01",
                "5dcee0eb-b34d-4652-acc3-d10afc6eae68_probe00",
                "d2832a38-27f6-452d-91d6-af72d794136c_probe00",
                "54238fd6-d2d0-4408-b1a9-d19d24fd29ce_probe00",
            ],
            "pickle_path": "/vast/th3129/data/ibl_new/spectrogram_preprocessed",
            "hc_acronyms": ["CA1", "CA2", "CA3", "DG", "VIS"],
        },
        "Neuronexus": {
            "sessions_list": [
                "AD_HF01_1",
                "AD_HF02_2",
                "AD_HF02_4",
                "AD_HF03_1",
                "AD_HF03_2",
                "NN_syn_01",
                "NN_syn_02",
            ],
            "pickle_path": "/scratch/th3129/region_decoding/data/Neuronexus/lfp",
            "hc_acronyms": ["CA1", "CA2", "CA3", "DG", "Cortex"],
        },
        "All": {
            "sessions_list": [
                "719161530",
                "794812542",
                "778998620",
                "798911424",
                "771990200",
                "771160300",
                "768515987",
            ],
            "test_sess": [
                "AD_HF01_1",
                "AD_HF02_4",
                "AD_HF03_1",
                "AD_HF03_2",
                "NN_syn_01",
                "NN_syn_02",
                "AD_HF02_2",
            ],
            "pickle_path": "spectrogram/Allen",
            "hc_acronyms": ["CA1", "CA2", "CA3", "DG", "VIS"],
        },
    }

    def __init__(
        self, data: str = "All", val_size: float = None, test_size: float = None
    ) -> None:
        if data not in self.DATA_SUITS:
            raise ValueError(f"Data {data} not found in DATA_SUITS")
        self.data = data
        self.sessions_list = self.DATA_SUITS[data]["sessions_list"]
        self.pickle_path = self.DATA_SUITS[data]["pickle_path"]
        self.hc_acronyms = self.DATA_SUITS[data]["hc_acronyms"]
        self.test_sess = self.DATA_SUITS[data].get("test_sess")
        self.val_size = val_size
        self.test_size = test_size

    def parse_datasets(
        self,
        session_list: Optional[list] = None,
        sampling_rate: Optional[int] = None,
    ) -> Tuple[LFP2VecDataset, Optional[LFP2VecDataset], Optional[LFP2VecDataset]]:
        """Load sessions, flatten, split, and optionally upsample waveforms.

        Returns train/val/test datasets (val/test may be None if sizes not set).
        """
        # if session_list is None, use self.sessions_list
        if session_list is None:
            session_list = self.sessions_list
        # else check if every session in session_list is in self.sessions_list
        else:
            for session in session_list:
                if session not in self.sessions_list:
                    raise ValueError(
                        f"Session {session} not found in self.sessions_list for {self.data}"
                    )

        features_dict, labels_dict, trials_dict, chans_dict = (
            self.load_preprocessed_data(
                self.pickle_path,
                session_list,
                "raw",
            )
        )

        # Flatten samples across sessions and build per-sample trial_chan identifiers
        all_data_parts = []
        all_label_parts = []
        all_trial_chans = []
        for session in session_list:
            session_features = features_dict[session]
            session_labels = labels_dict[session]
            session_trials = trials_dict[session]
            session_chans = chans_dict[session]

            all_data_parts.append(session_features)
            all_label_parts.append(session_labels)
            all_trial_chans.extend(
                [
                    f"{session}_{str(trial)}_{str(chan)}"
                    for trial, chan in zip(session_trials, session_chans)
                ]
            )

        data_all = (
            np.concatenate(all_data_parts, axis=0)
            if len(all_data_parts) > 0
            else np.array([])
        )
        labels_all = (
            np.concatenate(all_label_parts, axis=0)
            if len(all_label_parts) > 0
            else np.array([])
        )

        # Split data, labels, and trial_chans together to keep them aligned
        train_dataset, val_dataset, test_dataset = None, None, None

        X_train, y_train, tc_train = data_all, labels_all, all_trial_chans
        if self.val_size is not None:
            X_train, X_val, y_train, y_val, tc_train, tc_val = train_test_split(
                X_train, y_train, tc_train, test_size=self.val_size, random_state=42
            )
            val_dataset = LFP2VecDataset(X_val, y_val, tc_val)
        if self.test_size is not None:
            X_train, X_test, y_train, y_test, tc_train, tc_test = train_test_split(
                X_train, y_train, tc_train, test_size=self.test_size, random_state=42
            )
            test_dataset = LFP2VecDataset(X_test, y_test, tc_test)

        train_dataset = LFP2VecDataset(X_train, y_train, tc_train)

        if sampling_rate is not None:
            train_dataset.upsample_data(sampling_rate)
            val_dataset.upsample_data(sampling_rate)
            test_dataset.upsample_data(sampling_rate)

        return train_dataset, val_dataset, test_dataset

    def load_preprocessed_data(
        self,
        pickle_path: str,
        session_list: list,
        data_type: str = "raw",
    ):
        """Read per-session pickles and return dicts of features/labels/trials/chans."""
        features, labels, trials, chans = {}, {}, {}, {}
        for session in session_list:
            if data_type == "raw":
                data = pickle.load(open(f"{pickle_path}/{session}_raw.pickle", "rb"))
            elif data_type == "lfp":
                data = pickle.load(open(f"{pickle_path}/{session}_lfp.pickle", "rb"))
            X, y, trial_idx, chan_id = zip(*[(d[0], d[1], d[2], d[3]) for d in data])
            features[session] = np.array(X)
            non_zero_indices = [
                i for i, x in enumerate(features[session]) if not np.all(x == 0)
            ]
            features[session] = features[session][non_zero_indices]
            labels[session] = np.array(y, dtype=int)[non_zero_indices]
            trials[session] = np.array(trial_idx)[non_zero_indices]
            chans[session] = np.array(chan_id)[non_zero_indices]

            # Sanity check
            assert (
                len(features[session])
                == len(labels[session])
                == len(trials[session])
                == len(chans[session])
            ), f"Inconsistent data sizes for session {session}"

        return features, labels, trials, chans
