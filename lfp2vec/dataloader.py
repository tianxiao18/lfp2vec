import pickle
from typing import Optional, Tuple

import cupy as cp
import numpy as np
import torch
from cupyx.scipy.signal import resample as cupy_resample
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import random


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
        print(np.array(self.data).shape, np.array(self.labels).shape, np.array(self.chans).shape)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        return self.data[index], self.labels[index]

    def get_trial_chan(self, index: int) -> str:
        return self.chans[index]

    def upsample_data(self, signal_sampling_rate: int, target_sampling_rate: int = 16000) -> None:
        """Resample each 1D signal to 16 kHz and z-score normalize per sample."""
        epsilon = 1e-10
        upsampled_signals = []

        for signal in self.data:
            num_target_samples = target_sampling_rate
            if torch.cuda.is_available():
                signal = cp.asarray(signal)
                upsampled_signal = cupy_resample(signal, num_target_samples)
                upsampled_signal = upsampled_signal.get()
            else:
                upsampled_signal = resample(signal, num_target_samples)
            upsampled_signal = (upsampled_signal - np.mean(upsampled_signal)) / (
                np.std(upsampled_signal) + epsilon
            )
            upsampled_signals.append(upsampled_signal)

        self.data = np.array(upsampled_signals)


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
            "pickle_path": "/scratch/th3129/region_decoding/data/Allen/",
            "hc_acronyms": ["CA1", "CA2", "CA3", "DG", "VIS"],
            "test_sess": "719161530"
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
        "Monkey": {
            "sessions_list": [
                "221007",
                "221104",
                "221216"
            ],
            "pickle_path": "/scratch/th3129/lfp2vec/data/Monkey",
            "hc_acronyms": ["Basal_Ganglia", "Suppl_Motor_Area", "Primary_Motor_Cortex"],
            "test_sess": "221007"
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
        self, data: str = "All", train_session_size: float = 0.8, val_trial_size: float = 0.25, trial_length: int = 60,
    ) -> None:
        if data not in self.DATA_SUITS:
            raise ValueError(f"Data {data} not found in DATA_SUITS")
        self.data = data
        self.sessions_list = self.DATA_SUITS[data]["sessions_list"]
        self.pickle_path = self.DATA_SUITS[data]["pickle_path"]
        self.hc_acronyms = self.DATA_SUITS[data]["hc_acronyms"]
        self.test_sess = self.DATA_SUITS[data].get("test_sess")
        self.train_sess, self.val_sess = self.train_test_split_sessions(self.sessions_list, train_ratio=train_session_size)
        self.train_trials, self.val_trials, self.test_trials = self.train_test_split_trials(np.arange(trial_length), val_ratio=val_trial_size)

    def parse_datasets(
        self,
        session_list: Optional[list] = None,
        sampling_rate: Optional[int] = None,
    ) -> Tuple[LFP2VecDataset, LFP2VecDataset, LFP2VecDataset]:
        
        train_dataset = self._build_dataset(self.train_sess, self.train_trials, sampling_rate)
        val_dataset = self._build_dataset(self.val_sess, self.val_trials, sampling_rate)
        test_dataset = self._build_dataset([self.test_sess], self.test_trials, sampling_rate)
        
        return train_dataset, val_dataset, test_dataset

    def train_test_split_sessions(self, session_list: list, train_ratio: float, random_state: int=42):
        random.seed(random_state)

        if len(session_list) > 3:
            session_list.remove(self.test_sess)
            train_session_list = random.sample(session_list, int(len(session_list) * train_ratio))
            val_session_list = [ses for ses in session_list if ses not in train_session_list]
        else:
            train_session_list = [s for s in session_list if s != self.test_sess]
            val_session_list = [session_list[session_list.index(self.test_sess) - 1]] # select the session before test as validation session (with wrap around)
        
        return train_session_list, val_session_list

    def train_test_split_trials(self, trial_list: list, val_ratio: float, minimum_test_count=12, random_state: int=42):
        train_tr_idx, test_tr_idx = train_test_split(range(len(trial_list)), test_size=float(minimum_test_count/len(trial_list)), random_state=random_state)

        if len(trial_list) >= minimum_test_count + 4:
            train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=val_ratio, random_state=random_state)
        else:
            train_tr_idx, val_tr_idx = train_tr_idx, train_tr_idx

        return train_tr_idx, val_tr_idx, test_tr_idx

    def _build_dataset(
        self,
        session_list: Optional[list] = None,
        trial_list: Optional[list] = None,
        sampling_rate: Optional[int] = None,
    ) -> LFP2VecDataset:
        """Load sessions, flatten, split, and optionally upsample waveforms.

        Returns dataset of selected sessions and trials. 
        """
        # if session_list is None, use self.sessions_list
        if session_list is None:
            session_list = self.sessions_list
        # else check if every session in session_list is in self.sessions_list
        else:
            for session in session_list:
                if session not in self.sessions_list and session != self.test_sess:
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
            trial_idx = [idx for idx, val in enumerate(trials_dict[session]) if val in trial_list]

            session_features = features_dict[session][trial_idx]
            session_features = np.array([f if f.shape[0] == 3749 else f[:3749] for f in session_features])
            session_labels = labels_dict[session][trial_idx]
            session_trials = trials_dict[session][trial_idx]
            session_chans = chans_dict[session][trial_idx]

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

        dataset = LFP2VecDataset(data_all, labels_all, all_trial_chans)

        if sampling_rate is not None:
            dataset.upsample_data(sampling_rate)

        return dataset

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
                data = pickle.load(
                    open(f"{pickle_path}/lfp/{session}_raw.pickle", "rb")
                )
            elif data_type == "lfp":
                data = pickle.load(
                    open(f"{pickle_path}/lfp/{session}_lfp.pickle", "rb")
                )
            X, y, trial_idx, chan_id = zip(*[(d[0], d[1], d[2], d[3]) for d in data])
            y = [s.replace("imec", "") if isinstance(s, str) else s for s in y]
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
