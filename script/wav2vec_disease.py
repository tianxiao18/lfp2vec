import argparse
import os
import gc
import pickle
import random
import torch
import string
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import sys
import wandb
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.signal import resample
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence, Value
from transformers import AutoConfig, Wav2Vec2Config, AutoFeatureExtractor, Wav2Vec2ForPreTraining, AutoModelForAudioClassification, TrainingArguments, Trainer, pipeline, Wav2Vec2ForSequenceClassification
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from blind_localization.data.PCAviz import PCAVisualizer
from tqdm import tqdm
from matplotlib.patches import Wedge
from scipy.special import softmax
import matplotlib
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import seaborn as sns

def _sample_negatives(features: torch.FloatTensor, num_negatives: int):
        """
        Sample `num_negatives` vectors from feature vectors.
        """
        batch_size, sequence_length, hidden_size = features.shape
        if sequence_length <= 1:
            raise ValueError(
                f"`features should have `sequence_length` > 1, but are of shape (batch_size, sequence_length, hidden_size) = ({batch_size, sequence_length, hidden_size})."
            )

        features = features.view(-1, hidden_size)  # BTC => (BxT)C

        with torch.no_grad():
            # get `num_negatives` random vector indices from the same utterance
            sampled_negative_indices = torch.randint(
                low=0,
                high=sequence_length - 1,
                size=(batch_size, num_negatives * sequence_length),
                device=features.device,
            )

            # generate indices of the positive vectors themselves, repeat them `num_negatives` times
            feature_indices = (
                torch.arange(sequence_length, device=features.device)[:, None]
                .expand(sequence_length, num_negatives)
                .flatten()
            )

            # avoid sampling the same positive vector, but keep the distribution uniform
            sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1

        # correct for batch size
        for batch_idx in range(1, batch_size):
            sampled_negative_indices[batch_idx] += batch_idx * sequence_length

        # take negative vectors from sampled indices
        sampled_negatives = features[sampled_negative_indices.view(-1)]
        sampled_negatives = sampled_negatives.view(batch_size, sequence_length, num_negatives, hidden_size).permute(
            2, 0, 1, 3
        )

        return sampled_negatives


# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='wave2vec2')
    parser.add_argument('--data', type=str, default='Neuronexus')
    parser.add_argument('--trial_length', type=int, default=60, help='trial_length')
    parser.add_argument('--data_type', type=str, help='Data type to use', default='spectrogram_preprocessed')
    parser.add_argument('--sampling_rate', type=str, help='Sampling rate of the data', default='1250')
    parser.add_argument('--load_data', type=lambda x: x.lower() == 'true', help='Load data from disk or compute on fly', default=True)
    parser.add_argument('--rand_init', type=lambda x: x.lower() == 'true', help='random init or start from pretrained', default=False)
    parser.add_argument('--ssl', type=lambda x: x.lower() == 'true', help='self supervised training or fine tuning only', default=False)
    parser.add_argument('--session', type=str, help='session run or full run', default=None)
    return parser.parse_args()


args = arg_parser()
data, trial_length, data_type, sampling_rate = args.data, args.trial_length, args.data_type, args.sampling_rate, 
load_data, rand_init, ssl, selected_session = args.load_data, args.rand_init, args.ssl, args.session
print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}, Sampling Rate: {sampling_rate}")
print(f"Load Data: {load_data}, rand_init: {rand_init}, ssl: {ssl}, session: {selected_session}")
print("cuda is available: ", torch.cuda.is_available())

output_path = f"../results/{data}/{data_type}/wave2vec2/across_session"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# data_loading_path = f'/scratch/mkp6112/LFP/region_decoding/results/ibl/spectrogram/wave2vec2/across_session'
data_loading_path = output_path


def load_preprocessed_data(pickle_path, file_path, data_type='raw'):
    features, labels, trials, chans = {}, {}, {}, {}
    for session in file_path:
        if data_type == 'raw':
            data = pickle.load(open(f"{pickle_path}/{session}_full_raw.pickle", 'rb'))
        elif data_type == 'lfp':
            data = pickle.load(open(f"{pickle_path}/{session}_full_lfp.pickle", 'rb'))
        X, y, trial_idx, chan_id = zip(*[(d[0], d[1], d[2], d[3]) for d in data])
        # X = [x if x.shape[0] == 3749 else x[:3749] for x in X]s
        features[session] = np.array(X)
        labels[session] = np.array(y, dtype=int)
        trials[session] = np.array(trial_idx)
        chans[session] = np.array(chan_id)

        non_zero_indices = [i for i, x in enumerate(features[session]) if not np.all(x == 0)]

        features[session] = features[session][non_zero_indices]
        labels[session] = labels[session][non_zero_indices]
        trials[session] = trials[session][non_zero_indices]
        chans[session] = chans[session][non_zero_indices]

        # Sanity check
        assert len(features[session]) == len(labels[session]) == len(trials[session]) == len(chans[session]), \
            f"Inconsistent data sizes for session {session}"

    return features, labels, trials, chans


def preprocess_data(signals, signal_sampling_rate):
    target_sampling_rate = 16000
    epsilon = 1e-10
    upsampled_signals = []

    for signal in signals:
        # Calculate the number of samples for the target sampling rate
        num_target_samples = int(len(signal) * target_sampling_rate / signal_sampling_rate)
        # Resample the signal
        upsampled_signal = resample(signal, num_target_samples)
        upsampled_signal = (upsampled_signal - np.mean(upsampled_signal)) / (np.std(upsampled_signal) + epsilon)
        upsampled_signals.append(upsampled_signal)

    return np.array(upsampled_signals)

# def create_pretraining_dataset_dict(X_train, X_val, X_test):
#     train_dict = {"input_values": X_train.tolist()}
#     val_dict = {"input_values": X_val.tolist()}
#     test_dict = {"input_values": X_test.tolist()}

#     dataset_dict = DatasetDict({
#         "train": Dataset.from_dict(train_dict),
#         "validation": Dataset.from_dict(val_dict),
#         "test": Dataset.from_dict(test_dict),
#     })

#     return dataset_dict

def create_dataset_dict(X_train, y_train, X_val, y_val, X_test, y_test, custom_features):
    # Create individual datasets
    train_dict = {"label": y_train, "input_values": X_train.tolist() if type(X_train) is not list else X_train}
    val_dict = {"label": y_val, "input_values": X_val.tolist() if type(X_val) is not list else X_val}
    test_dict = {"label": y_test, "input_values": X_test.tolist() if type(X_test) is not list else X_test}
    
    # Combine into a DatasetDict
    print("Combining datasets...")
    dataset_dict = DatasetDict({
        "train": Dataset.from_dict(train_dict, features=custom_features),
        "validation": Dataset.from_dict(val_dict, features=custom_features),
        "test": Dataset.from_dict(test_dict, features=custom_features)
    })

    return dataset_dict


def compute_mask_inputs(model, input_values, device):
    batch_size, raw_seq_len = input_values.shape
    with torch.no_grad():
        # Compute the feature extractor output length
        seq_len = model._get_feat_extract_output_lengths(raw_seq_len).item()
        # Compute masking
        mask_time_indices = _compute_mask_indices(
            (batch_size, seq_len),
            mask_prob=model.config.mask_time_prob,
            mask_length=model.config.mask_time_length
        )
        # print(mask_time_indices.shape, mask_time_indices.sum(), model.config.mask_time_prob, model.config.mask_time_length)
        assert(mask_time_indices.sum() > 0)
        sampled_negatives = _sample_negative_indices(
            (batch_size, seq_len),
            num_negatives=model.config.num_negatives,
            mask_time_indices=mask_time_indices
        )
        mask_time_indices = torch.tensor(mask_time_indices).to(device)
        sampled_negatives = torch.tensor(sampled_negatives).to(device)
    return mask_time_indices, sampled_negatives

def train(model, train_loader, optimizer, device):
        total_loss = 0
        grad_norms = []
        model.train()
        for (input_values, ) in train_loader:
            input_values = input_values.float().to(device)
            mask_time_indices, sampled_negative_indices = compute_mask_inputs(model, input_values, device)
            sampled_negative_indices = sampled_negative_indices.to(device)

            outputs = model(input_values=input_values,
                mask_time_indices=mask_time_indices,
                sampled_negative_indices=sampled_negative_indices
            )

            loss = outputs.loss
            loss.backward()
            grad_norm = get_grad_norm(model)
            grad_norms.append(grad_norm)
            # print(f"loss: {loss}, grad_norm: {grad_norm}")
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_grad = sum(grad_norms) / len(grad_norms)
        return avg_loss, avg_grad
    
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for (input_values, ) in val_loader:
            input_values = input_values.float().to(device)
            mask_time_indices, sampled_negative_indices = compute_mask_inputs(model, input_values, device)

            outputs = model(input_values=input_values,
                mask_time_indices=mask_time_indices,
                sampled_negative_indices=torch.tensor(sampled_negative_indices).to(device)
            )

            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

def collect_embeddings(data, labels, trainer, custom_features, batch_size=30, split_name=""):
    embeddings = []
    collected_labels = []
    current_index = 0
    total_samples = len(data)

    while current_index < total_samples:
        print(f"{split_name} {current_index}/{total_samples}")

        # Get the current batch
        end_index = min(current_index + batch_size, total_samples)
        batch_data = data[current_index:end_index]
        batch_labels = labels[current_index:end_index]

        # Update index
        current_index = end_index

        # Create Hugging Face dataset
        batch_dict = {
            "label": batch_labels,
            "input_values": batch_data.tolist()
        }
        batch_dataset = Dataset.from_dict(batch_dict, features=custom_features)

        # Run prediction
        predictions = trainer.predict(batch_dataset.with_format("torch"))

        # Extract embeddings — assuming your model saves them to classifier_input_first
        temp_embeddings = classifier_input_first.cpu().detach().numpy()

        # Store embeddings and labels
        embeddings.extend(temp_embeddings)
        collected_labels.extend(batch_labels)

    return embeddings, collected_labels

def get_grad_norm(model, norm_type=2.0):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)

def plot(temp1, temp2):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(temp1, label='Original')
    plt.subplot(2, 1, 2)
    plt.plot(temp2, label='Upsampled')
    plt.legend()
    plt.savefig('upsampled_signal.png')


def projector_hook(module, input, output):
    global projector_output
    projector_output = output


def classifier_hook(module, input, output):
    global classifier_input_first, classifier_input_whole, classifier_output
    classifier_input_first = input[0]
    classifier_input_whole = input
    classifier_output = output

def quantizer_hook(module, input, output):
    global debug_data
    hidden_states = input[0]
    hidden_states = input[0]  # [B, T, D]
    batch_size, seq_len, hidden_size = hidden_states.shape

    proj = module.weight_proj(hidden_states)  # shape [B, T, groups * num_vars]
    logits = proj.view(batch_size * seq_len * module.num_groups, -1) # shape [B*T*G, num_vars]

    soft_probs = torch.softmax(
        logits.view(batch_size * seq_len, module.num_groups, -1).float(), dim=-1
    )

    debug_data['logits'] = logits.detach().cpu()
    debug_data['soft_probs'] = soft_probs.detach().cpu()
    

class LinearProber(nn.Module):
    def __init__(self, encoder, rep_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(rep_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            reps = self.encoder(x).last_hidden_state.detach()  # e.g., output shape: (B, T, D)
            reps = reps.mean(dim=1)  # first token pooling, also consider mean pooling
        return self.classifier(reps)

def train_probe(encoder, train_loader, val_loader, rep_dim, device):
    for p in encoder.parameters():
        p.requires_grad = False  # freeze encoder

    prober = LinearProber(encoder=encoder, rep_dim=rep_dim, num_classes=5).to(device)

    prober.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(prober.classifier.parameters(), lr=1e-4)
    train_loss, train_correct, val_loss, val_correct = 0, 0, 0, 0

    # training the model
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = prober(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        train_correct += (logits.argmax(1) == yb).sum().item()

    prober.eval()
    # validate the model
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = prober(xb)
        loss = criterion(logits, yb)
        val_loss += loss.item() * xb.size(0)
        val_correct += (logits.argmax(1) == yb).sum().item()

    train_avg_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)
    val_avg_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    for p in encoder.parameters():
        p.requires_grad = True

    return train_avg_loss, train_acc, val_avg_loss, val_acc

def split_trials(trial_length, fixed_test_count=12, val_size=0.25, random_state=42):
    if not isinstance(trial_length, int) or trial_length < fixed_test_count + 4:
        raise ValueError(f"trial_length must be greater than {fixed_test_count + 4}")

    print("trial_length:", trial_length)

    all_indices = np.arange(trial_length)
    test_tr_idx = np.random.RandomState(seed=random_state).choice(all_indices, size=fixed_test_count, replace=False)

    remaining_indices = np.setdiff1d(all_indices, test_tr_idx)
    train_tr_idx, val_tr_idx = train_test_split(remaining_indices, test_size=val_size, random_state=random_state)

    print("Train indices:", train_tr_idx)
    print("Validation indices:", val_tr_idx)
    print("Test indices:", test_tr_idx)

    return train_tr_idx, val_tr_idx, test_tr_idx

def session_to_disease(session):
    return int("AD" in session)

def run_wave2vec2(sessions, sess, sampling_rate, acronyms_arr):
    label_json = {str(i): region for i, region in enumerate(acronyms_arr)}
    print(label_json)

    session_list = sessions.copy()
    test_sessions = sess

    signal_type = 'lfp' if 'lfp' in pickle_path else 'raw'

    train_trial_chans, val_trial_chans, test_trial_chans = [], [], []

    print(f"Session list: {session_list}")
    test_features, test_labels, test_trials, test_chans = load_preprocessed_data(pickle_path, test_sessions, signal_type)
    features, labels, trials, chans = [], [], [], []
    for sessions in test_sessions:
        temp_features = test_features.get(sessions)
        # temp_features = [f if f.shape[0] == 3749 else f[:3749] for f in temp_features]
        temp_features = np.array(temp_features)

        features.extend(temp_features)
        trials.extend(test_trials.get(sessions))
        chans.extend(test_chans.get(sessions))

        temp_test_labels = test_labels.get(sessions)
        temp_test_labels = [session_to_disease(sessions) for label in temp_test_labels]
        labels.extend(temp_test_labels)

        temp_trials = test_trials.get(sessions)
        temp_test_chans = test_chans.get(sessions)
        temp_test_chans = [f"{sessions}_{str(temp_trials[i])}_{str(temp_test_chans[i])}" for i in
                           range(len(temp_trials))]
        test_trial_chans.extend(temp_test_chans)

    features = np.array(features)
    labels = np.array(labels)
    trials = np.array(trials)
    test_trial_chans = np.array(test_trial_chans)

    train_tr_idx, val_tr_idx, test_tr_idx = split_trials(trial_length)


    test_idx = [idx for idx, val in enumerate(trials) if val in test_tr_idx]

    X_train, y_train, X_val, y_val = [], [], [], []
    X_test, y_test = features[test_idx], labels[test_idx]
    test_trial_chans = test_trial_chans[test_idx]

    session_list = [s for s in session_list if s not in test_sessions]
    random.seed(11)
    train_session_list = random.sample(session_list, int(len(session_list) * 0.8))
    val_session_list = [ses for ses in session_list if ses not in train_session_list]
    print("Training on ", train_session_list)
    print("Validate on ", val_session_list)
    print("Testing on ", test_sessions)

    all_features, all_labels, all_trials, all_chans = load_preprocessed_data(pickle_path, session_list, signal_type)
    for sess in train_session_list:
        features = all_features.get(sess)
        # features = [f if f.shape[0] == 3749 else f[:3749] for f in features]
        features = np.array(features)
        trial_idx = all_trials.get(sess)
        chans = all_chans.get(sess)

        idx_train = [idx for idx, val in enumerate(trial_idx) if val in train_tr_idx]

        temp_train_labels = all_labels.get(sess)
        labels = np.array([session_to_disease(sess) for label in temp_train_labels])
        
        temp_train_trials = trial_idx[idx_train]
        temp_train_chans = chans[idx_train]
        temp_train_chans = [f"{sess}_{str(temp_train_trials[i])}_{str(temp_train_chans[i])}"
                            for i in range(len(temp_train_trials))]
        train_trial_chans.extend(temp_train_chans)

        X_train.extend(features[idx_train])
        y_train.extend(labels[idx_train])

    for sess in val_session_list:
        features = all_features.get(sess)
        # features = [f if f.shape[0] == 3749 else f[:3749] for f in features]
        features = np.array(features)
        labels = all_labels.get(sess)
        trial_idx = all_trials.get(sess)
        chans = all_chans.get(sess)

        idx_val = [idx for idx, val in enumerate(trial_idx) if val in val_tr_idx]
        
        temp_val_labels = all_labels.get(sess)
        labels = np.array([session_to_disease(sess) for label in temp_val_labels])

        temp_val_trials = trial_idx[idx_val]
        temp_val_chans = chans[idx_val]
        temp_val_chans = [f"{sess}_{str(temp_val_trials[i])}_{str(temp_val_chans[i])}"
                          for i in range(len(temp_val_trials))]
        val_trial_chans.extend(temp_val_chans)

        X_val.extend(features[idx_val])
        y_val.extend(labels[idx_val])

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    train_trial_chans = np.array(train_trial_chans)
    val_trial_chans = np.array(val_trial_chans)

    print(f"AD trials: {sum(y_train)}, NN trials: {len(y_train) - sum(y_train)}")

    y_train = [str(label) for label in y_train]
    y_val = [str(label) for label in y_val]
    y_test = [str(label) for label in y_test]

    assert len(y_train) == len(train_trial_chans), "Length of y_train and train_trial_chans are not equal"
    assert len(y_val) == len(val_trial_chans), "Length of y_val and val_trial_chans are not equal"
    assert len(y_train) == len(X_train), "Length of y_train and X_train are not equal"

    def calculate_chance_accuracy(y_labels):
        label_counts = {}
        for label in y_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        chance_accuracy = max(label_counts.values()) / sum(label_counts.values())
        return label_counts, chance_accuracy

    train_label_counts, train_chance_accuracy = calculate_chance_accuracy(y_train)
    val_label_counts, val_chance_accuracy = calculate_chance_accuracy(y_val)
    test_label_counts, test_chance_accuracy = calculate_chance_accuracy(y_test)

    # percent_to_stratify = 0.2
    # X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1-percent_to_stratify, stratify=y_train)
    # X_val, _, y_val, _ = train_test_split(X_val, y_val, test_size=1-percent_to_stratify, stratify=y_val)
    # X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=1-percent_to_stratify, stratify=y_test)


    id2label = {str(i): acr for i, acr in enumerate(acronyms_arr)}
    label2id = {acr: str(i) for i, acr in enumerate(acronyms_arr)}

    custom_features = Features({
        'label': ClassLabel(num_classes=len(acronyms_arr), names=acronyms_arr),
        'input_values': Sequence(feature=Value(dtype='float'), length=-1)
    })

    print(f"label2id: {label2id}")
    print(f"id2label: {id2label}")
    print(f"Features: {custom_features}")
    print(f"Unique labels in dataset: {np.unique(y_train + y_val + y_test)}")
    print(f"Expected labels in id2label: {list(id2label.keys())}")

    print("Upsampling the data...")
    print(X_train.shape, sampling_rate)
    X_train_upsampled = preprocess_data(X_train, sampling_rate)
    X_val_upsampled = preprocess_data(X_val, sampling_rate)
    X_test_upsampled = preprocess_data(X_test, sampling_rate)

    # temp1 = X_train[0]
    # temp2 = X_train_upsampled[0]
    # plot(temp1, temp2)

    print("Creating dataset dictionary...")
    # if load_data and os.path.exists(f"{data_loading_path}/{session}_{trial_length}_dataset_dict.pickle"):
    # if load_data and os.path.exists(f"{data_loading_path}/disease_dataset_dict.pickle"):
    #     print(f"Loading from {data_loading_path}/disease_dataset_dict.pickle")
    #     with open(f"{data_loading_path}/disease_dataset_dict.pickle", 'rb') as f:
    #         dataset_dict = pickle.load(f)
    # else:
    dataset_dict = create_dataset_dict(X_train_upsampled, y_train, X_val_upsampled, y_val, X_test_upsampled, y_test, custom_features)
    if load_data:
        with open(f"{output_path}/disease_dataset_dict.pickle", 'wb') as f:
            pickle.dump(dataset_dict, f)

    def generate_random_string(length):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    random_string = generate_random_string(6)
    unique_cache_dir = tempfile.mkdtemp(prefix="hf_eval_")
    accuracy = evaluate.load("accuracy", experiment_id=random_string, cache_dir=unique_cache_dir)

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

    num_labels = len(id2label)

    print("Training the model...")
    w2v2_config = {"vocab_size": 32, "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
                "intermediate_size": 768*4, "hidden_act": "gelu", "hidden_dropout": 0.1,
                "attention_dropout": 0.1, "final_dropout": 0.1, "initializer_range": 0.02,
                "layer_norm_eps": 1e-12, "feat_extract_norm": "group", "feat_proj_dropout": 0.0,
                "feat_extract_activation": "gelu", "feat_quantizer_dropout": 0.0,
                "conv_dim": (512, 512, 512, 512, 512, 512, 512),
                "conv_stride": (5, 2, 2, 2, 2, 2, 2), "conv_kernel": (10, 3, 3, 3, 3, 2, 2), "conv_bias": False,
                "num_conv_pos_embeddings": 128, "num_conv_pos_embeddings_groups": 16, "do_stable_layer_norm": False,
                "apply_spec_augment": True, "mask_time_prob": 0.05, "mask_time_length": 10, "mask_feature_prob": 0.0,
                "mask_feature_length": 10, "num_codevectors_per_group": 320, "num_codevector_groups": 2,
                "contrastive_logits_temperature": 0.1, "num_negatives": 100, "codevector_dim": 256,
                "proj_codevector_dim": 256, "diversity_loss_weight": 0.1, "ctc_loss_reduction": "sum",
                "ctc_zero_infinity": False, "use_weighted_layer_sum": False, "classifier_proj_size": 256}
    config = Wav2Vec2Config(**w2v2_config)
    config.update({
                    # 'num_negatives': 16, 'do_stable_layer_norm':True,
                    # 'contrastive_logits_temperature': 0.05, "gumbel_temperature": 0.5,
                    'mask_time_min_masks':2, 'mask_time_prob':0.2,
                    'random_init': rand_init,'self_supervised': ssl})
    train_config = {'epoch': 100}

    ri_tag = 'rand_init' if rand_init else 'pretrained'
    ssl_tag = 'ssl' if ssl else 'nossl'
    wandb.init(project="wav2vec_ssl", config=w2v2_config, name=f"disease-{ri_tag}-{ssl_tag}-exp", reinit=True)
    # wandb.log({"percent_data": percent_to_stratify})
    # w2v2_config = {"vocab_size": 32, "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
    #                "intermediate_size": 3072, "hidden_act": "gelu", "hidden_dropout": 0.1,
    #                "attention_dropout": 0.1, "final_dropout": 0.1, "initializer_range": 0.02,
    #                "layer_norm_eps": 1e-12, "feat_extract_norm": "group", "feat_proj_dropout": 0.0,
    #                "feat_extract_activation": "gelu", "feat_quantizer_dropout": 0.0,
    #                "conv_dim": (512, 512, 512, 512, 512, 512, 512),
    #                "conv_stride": (5, 2, 2, 2, 2, 2, 2), "conv_kernel": (10, 3, 3, 3, 3, 2, 2), "conv_bias": False,
    #                "num_conv_pos_embeddings": 128, "num_conv_pos_embeddings_groups": 16, "do_stable_layer_norm": False,
    #                "apply_spec_augment": True, "mask_time_prob": 0.05, "mask_time_length": 10, "mask_feature_prob": 0.0,
    #                "mask_feature_length": 10, "num_codevectors_per_group": 320, "num_codevector_groups": 2,
    #                "contrastive_logits_temperature": 0.1, "num_negatives": 100, "codevector_dim": 256,
    #                "proj_codevector_dim": 256, "diversity_loss_weight": 0.1, "ctc_loss_reduction": "sum",
    #                "ctc_zero_infinity": False, "use_weighted_layer_sum": False, "classifier_proj_size": 256}
    # config = Wav2Vec2Config(**w2v2_config)
    if rand_init:
        ssl_model = Wav2Vec2ForPreTraining(config=config)
    else:
        ssl_model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", config=config)

    ssl_model.quantizer.register_forward_hook(quantizer_hook)
    ssl_model = ssl_model.train()

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_model.to(device)
    ssl_model.save_pretrained(f"{output_path}/disease/ssl_model/")
    print(f"Model is on device: {ssl_model.device}")
    print(X_train_upsampled.shape, X_val_upsampled.min(), X_test_upsampled.max())

    # self supervised pretraining
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=1e-4)
    train_dataset = TensorDataset(torch.tensor(X_train_upsampled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_upsampled, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_upsampled, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    train_probe_dataset = TensorDataset(torch.tensor(X_train_upsampled, dtype=torch.float32), torch.tensor(np.array(y_train).astype(int)))
    val_probe_dataset = TensorDataset(torch.tensor(X_val_upsampled, dtype=torch.float32), torch.tensor(np.array(y_val).astype(int)))
    train_probe_loader = DataLoader(train_probe_dataset, batch_size=32, shuffle=True)
    val_probe_loader = DataLoader(val_probe_dataset, batch_size=32, shuffle=True)
    max_probe_acc = 0

    if ssl:
        for epoch in tqdm(range(train_config['epoch'])):
            train_loss, grad_norm = train(ssl_model, train_loader, optimizer, device)
            val_loss = validate(ssl_model, val_loader, device)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Grad Norm: {grad_norm:.4f}")
            wandb.log({"Train Loss": train_loss, "Val Loss": val_loss, "Grad Norm": grad_norm, "learning_rate": optimizer.param_groups[0]['lr']})

            if (epoch + 1) % 1 == 0:
                probe_train_loss, probe_train_acc, probe_val_loss, probe_val_acc = train_probe(ssl_model.wav2vec2, train_probe_loader, val_probe_loader, w2v2_config["hidden_size"], device)
                print(f"Probe Train Loss: {probe_train_loss:.4f}, Probe Train Accuracy: {probe_train_acc:.4f}, ")
                print(f"Probe Val Loss: {probe_val_loss:.4f}, Probe Val Accuracy: {probe_val_acc:.4f}")
                wandb.log({"Probe Train Loss": probe_train_loss, "Probe Train Accuracy": probe_train_acc,
                            "Probe Val Loss": probe_val_loss, "Probe Val Accuracy": probe_val_acc})

                if max_probe_acc > probe_val_acc:
                    max_probe_acc = max(max_probe_acc, probe_val_acc)
                    ssl_model.save_pretrained(f"{output_path}/disease/ssl_model/")

    # supervised fine tuning
    # w2v2_config = Wav2Vec2Config(**w2v2_config)
    # w2v2_config.num_labels = num_labels
    # w2v2_config.label2id = label2id
    # w2v2_config.id2label = id2label

    # model = Wav2Vec2ForSequenceClassification(w2v2_config)
    ssl_model = Wav2Vec2ForPreTraining.from_pretrained(f"{output_path}/disease/ssl_model/")
    # w2v2_config = {"vocab_size": 32, "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
    #                "intermediate_size": 3072, "hidden_act": "gelu", "hidden_dropout": 0.1,
    #                "attention_dropout": 0.1, "final_dropout": 0.1, "initializer_range": 0.02,
    #                "layer_norm_eps": 1e-12, "feat_extract_norm": "group", "feat_proj_dropout": 0.0,
    #                "feat_extract_activation": "gelu", "feat_quantizer_dropout": 0.0,
    #                "conv_dim": (512, 512, 512, 512, 512, 512, 512),
    #                "conv_stride": (5, 2, 2, 2, 2, 2, 2), "conv_kernel": (10, 3, 3, 3, 3, 3, 3), "conv_bias": False,
    #                "num_conv_pos_embeddings": 128, "num_conv_pos_embeddings_groups": 16, "do_stable_layer_norm": False,
    #                "apply_spec_augment": True, "mask_time_prob": 0.05, "mask_time_length": 10, "mask_feature_prob": 0.0,
    #                "mask_feature_length": 10, "num_codevectors_per_group": 320, "num_codevector_groups": 2,
    #                "contrastive_logits_temperature": 0.1, "num_negatives": 100, "codevector_dim": 256,
    #                "proj_codevector_dim": 256, "diversity_loss_weight": 0.1, "ctc_loss_reduction": "sum",
    #                "ctc_zero_infinity": False, "use_weighted_layer_sum": False, "classifier_proj_size": 256}
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label, config=w2v2_config
    )
    model.wav2vec2.load_state_dict(ssl_model.wav2vec2.state_dict())

    training_args = TrainingArguments(
        output_dir=f"{output_path}/disease",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"].with_format("torch"),
        eval_dataset=dataset_dict["train"].with_format("torch"),
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
    )

    # first visualize the embeddings before fine tuning
    projector_hook_handle = model.projector.register_forward_hook(projector_hook)
    classifier_hook_handle = model.classifier.register_forward_hook(classifier_hook)

    # train_embeddings, train_labels = collect_embeddings(X_train_upsampled, y_train, trainer, custom_features, split_name="Train")
    # val_embeddings, val_labels = collect_embeddings(X_val_upsampled, y_val, trainer, custom_features, split_name="Val")
    # test_embeddings, test_labels = collect_embeddings(X_test_upsampled, y_test, trainer, custom_features, split_name="Test")

    # ft_tag = "no_ft"
    # visualizer = PCAVisualizer(label_json, output_path=output_path)
    # try:
    #     visualizer.create_pca(train_embeddings, train_labels, 2, f"{session}_{ri_tag}_{ssl_tag}_{ft_tag}", "train")
    #     visualizer.create_pca(train_embeddings, train_labels, 3, f"{session}_{ri_tag}_{ssl_tag}_{ft_tag}", "train")
    #     visualizer.create_pca(val_embeddings, val_labels, 2, f"{session}_{ri_tag}_{ssl_tag}_{ft_tag}", "val")
    #     visualizer.create_pca(val_embeddings, val_labels, 3, f"{session}_{ri_tag}_{ssl_tag}_{ft_tag}", "val")
    #     visualizer.create_pca(test_embeddings, test_labels, 2, f"{session}_{ri_tag}_{ssl_tag}_{ft_tag}", "test")
    #     visualizer.create_pca(test_embeddings, test_labels, 3, f"{session}_{ri_tag}_{ssl_tag}_{ft_tag}", "test")

    #     datasets = {
    #         "train": {"embeddings": train_embeddings, "labels": train_labels},
    #         "val": {"embeddings": val_embeddings, "labels": val_labels},
    #         "test": {"embeddings": test_embeddings, "labels": test_labels}
    #     }
    #     visualizer.create_combined_pca(datasets, 2, f"{session}_{ri_tag}_{ssl_tag}_{ft_tag}", "combined")
    #     visualizer.create_combined_pca(datasets, 3, f"{session}_{ri_tag}_{ssl_tag}_{ft_tag}", "combined")
    # except:
    #     print(f"Possible size mismatch: "
    #         f"Train({len(train_embeddings)}, {len(train_labels)}), "
    #         f"Val({len(val_embeddings)}, {len(val_labels)}), "
    #         f"Test({len(test_embeddings)}, {len(test_labels)})")
    
    # then fine tune the model here
    best_ckpt_path = None
    trainer.train()
    best_ckpt_path = trainer.state.best_model_checkpoint

    # Pickle relevant data
    train_pred = trainer.predict(dataset_dict["train"].with_format("torch"))
    train_logits = train_pred[0]
    train_labels = train_pred[1]
    train_acc = train_pred[2]['test_accuracy']

    val_pred = trainer.predict(dataset_dict["validation"].with_format("torch"))
    val_logits = val_pred[0]
    val_labels = val_pred[1]
    val_acc = val_pred[2]['test_accuracy']

    test_pred = trainer.predict(dataset_dict["test"].with_format("torch"))
    test_logits = test_pred[0]
    test_labels = test_pred[1]
    test_acc = test_pred[2]['test_accuracy']
    
    train_embeddings, train_labels = collect_embeddings(X_train_upsampled, y_train, trainer, custom_features, split_name="Train")
    val_embeddings, val_labels = collect_embeddings(X_val_upsampled, y_val, trainer, custom_features, split_name="Val")
    test_embeddings, test_labels = collect_embeddings(X_test_upsampled, y_test, trainer, custom_features, split_name="Test")

    ft_tag = "ft"
    visualizer = PCAVisualizer(label_json, output_path=output_path)
    try:
        visualizer.create_pca(train_embeddings, train_labels, 2, f"disease_{ri_tag}_{ssl_tag}_{ft_tag}", "train")
        visualizer.create_pca(train_embeddings, train_labels, 3, f"disease_{ri_tag}_{ssl_tag}_{ft_tag}", "train")
        visualizer.create_pca(val_embeddings, val_labels, 2, f"disease_{ri_tag}_{ssl_tag}_{ft_tag}", "val")
        visualizer.create_pca(val_embeddings, val_labels, 3, f"disease_{ri_tag}_{ssl_tag}_{ft_tag}", "val")
        visualizer.create_pca(test_embeddings, test_labels, 2, f"disease_{ri_tag}_{ssl_tag}_{ft_tag}", "test")
        visualizer.create_pca(test_embeddings, test_labels, 3, f"disease_{ri_tag}_{ssl_tag}_{ft_tag}", "test")

        datasets = {
            "train": {"embeddings": train_embeddings, "labels": train_labels},
            "val": {"embeddings": val_embeddings, "labels": val_labels},
            "test": {"embeddings": test_embeddings, "labels": test_labels}
        }
        visualizer.create_combined_pca(datasets, 2, f"disease_{ri_tag}_{ssl_tag}_{ft_tag}", "combined")
        visualizer.create_combined_pca(datasets, 3, f"disease_{ri_tag}_{ssl_tag}_{ft_tag}", "combined")
    except:
        print(f"Possible size mismatch: "
            f"Train({len(train_embeddings)}, {len(train_labels)}), "
            f"Val({len(val_embeddings)}, {len(val_labels)}), "
            f"Test({len(test_embeddings)}, {len(test_labels)})")

    file_path = os.path.join(output_path, f'disease_{ri_tag}_{ssl_tag}_{random_string}_results.pickle')
    file_obj = {
        'train_logits': train_logits,  # Predicted logits for training set
        'train_labels': train_labels,  # True labels for training set
        'train_acc': train_acc,  # Accuracy for training set
        'train_embeddings': train_embeddings,  # Projector layer Embeddings for training set
        'train_trial_chans': train_trial_chans,  # Trial and channel IDs for training set ("<Session>_<Trial>_<Chan>")
        'val_logits': val_logits,  # Predicted logits for validation set
        'val_labels': val_labels,  # True labels for validation set
        'val_acc': val_acc,  # Accuracy for validation set
        'val_embeddings': val_embeddings,  # Projector layer Embeddings for validation set
        'val_trial_chans': val_trial_chans,  # Trial and channel IDs for validation set ("<Session>_<Trial>_<Chan>")
        'test_logits': test_logits,  # Predicted logits for test set
        'test_labels': test_labels,  # True labels for test set
        'test_acc': test_acc,  # Accuracy for test set
        'test_embeddings': test_embeddings,  # Projector layer Embeddings for test set
        'test_trial_chans': test_trial_chans,  # Trial and channel IDs for test set ("<Session>_<Trial>_<Chan>")
        'train_label_counts': train_label_counts,  # Label counts for training set
        'train_chance_accuracy': train_chance_accuracy,  # Chance accuracy for training set
        'val_label_counts': val_label_counts,  # Label counts for validation set
        'val_chance_accuracy': val_chance_accuracy,  # Chance accuracy for validation set
        'test_label_counts': test_label_counts,  # Label counts for test set
        'test_chance_accuracy': test_chance_accuracy,  # Chance accuracy for test set
        'w2v2_config': w2v2_config,  # Model configuration
        'best_ckpt_path': best_ckpt_path  # Path to the best checkpoint
    }
    with open(file_path, 'wb') as f:
        pickle.dump(file_obj, f)
        print(f"Disease results saved to {file_path}")
    print(f"Train acc: {file_obj['train_acc']}, Val acc: {file_obj['val_acc']}, Test acc: {file_obj['test_acc']}")
    wandb.log({"Train acc": file_obj['train_acc'], "Val acc": file_obj['val_acc'], "Test acc": file_obj['test_acc']})

    # Deregister hooks
    projector_hook_handle.remove()
    classifier_hook_handle.remove()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if data == "Neuronexus":
        sessions_list = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02']
        pickle_path = f'/scratch/th3129/region_decoding/data/Neuronexus/lfp'
        hc_acronyms = ['NN', 'AD']

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    to_skip = []
    test_sess = [random.choice(sessions_list[:5])] + [random.choice(sessions_list[5:])]

    # Hooks
    projector_output = None
    classifier_input_first = None
    classifier_input_whole = None
    classifier_output = None
    debug_data = {}

    run_wave2vec2(sessions_list, test_sess, int(sampling_rate), hc_acronyms)