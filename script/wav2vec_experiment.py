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
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.signal import resample
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence, Value
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer, pipeline
from blind_localization.data.PCAviz import PCAVisualizer
from tqdm import tqdm
from matplotlib.patches import Wedge
from scipy.special import softmax
import matplotlib
from scipy.signal import butter, filtfilt

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='wave2vec2')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen or ibl', default='Allen')
    parser.add_argument('--trial_length', type=int, default=60, help='trial_length')
    parser.add_argument('--data_type', type=str, help='Data type to use', default='spectrogram_preprocessed')
    parser.add_argument('--sampling_rate', type=str, help='Sampling rate of the data', default='1250')
    parser.add_argument('--load_data', type=lambda x: x.lower() == 'true', help='Load data from disk or compute on fly', default=True)
    return parser.parse_args()


args = arg_parser()
data, trial_length, data_type, sampling_rate, load_data = args.data, args.trial_length, args.data_type, args.sampling_rate, args.load_data
print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}, Sampling Rate: {sampling_rate}")
print(torch.cuda.is_available())

output_path = f"../results/{data}/{data_type}/wave2vec2/across_session"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# data_loading_path = f'/scratch/mkp6112/LFP/region_decoding/results/ibl/spectrogram/wave2vec2/across_session'
data_loading_path = output_path


def load_preprocessed_data(pickle_path, file_path, data_type='raw'):
    features, labels, trials, chans = {}, {}, {}, {}
    for session in file_path:
        if data_type == 'raw':
            data = pickle.load(open(f"{pickle_path}/{session}_raw.pickle", 'rb'))
        elif data_type == 'lfp':
            data = pickle.load(open(f"{pickle_path}/{session}_lfp.pickle", 'rb'))
        X, y, trial_idx, chan_id = zip(*[(d[0], d[1], d[2], d[3]) for d in data])
        X = [x if x.shape[0] == 3749 else x[:3749] for x in X]
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


def preprocess_data(signals, signal_sampling_rate, freq_band=[30, 100]):
    target_sampling_rate = 16000
    epsilon = 1e-10
    upsampled_signals = []

    for signal in signals:
        # Calculate the number of samples for the target sampling rate
        selected_signal = bandpass_filter(signal, fs=signal_sampling_rate, lowcut=freq_band[0], highcut=freq_band[1])

        num_target_samples = int(len(signal) * target_sampling_rate / signal_sampling_rate)

        # Resample the signal
        upsampled_signal = resample(selected_signal, num_target_samples)

        if args.data == 'All':
            upsampled_signal = (upsampled_signal - np.mean(upsampled_signal)) / (np.std(upsampled_signal) + epsilon)

        upsampled_signals.append(upsampled_signal)

    return np.array(upsampled_signals)


def create_dataset_dict(X_train, y_train, X_val, y_val, X_test, y_test, custom_features):
    # Create individual datasets
    train_dict = {"label": y_train, "input_values": X_train.tolist() if type(X_train) is not list else X_train}
    # print(X_train.shape)
    train_dataset = Dataset.from_dict(train_dict, features=custom_features)

    val_dict = {"label": y_val, "input_values": X_val.tolist() if type(X_val) is not list else X_val}
    # print(X_val.shape)
    val_dataset = Dataset.from_dict(val_dict, features=custom_features)

    test_dict = {"label": y_test, "input_values": X_test.tolist() if type(X_test) is not list else X_test}
    # print(X_test.shape)
    test_dataset = Dataset.from_dict(test_dict, features=custom_features)

    print("Combining datasets...")
    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    return dataset_dict


def plot(temp1, temp2, title):
    plt.figure(figsize=(10, 6))
    plt.plot(temp1, label='Original')
    plt.plot(temp2, label='Upsampled')
    plt.legend()
    plt.savefig(title)


def projector_hook(module, input, output):
    global projector_output
    projector_output = output


def classifier_hook(module, input, output):
    global classifier_input_first, classifier_input_whole, classifier_output
    classifier_input_first = input[0]
    classifier_input_whole = input
    classifier_output = output


def run_wave2vec2(sessions, sess, sampling_rate, acronyms_arr, freq_band):
    label_json = {str(i): region for i, region in enumerate(acronyms_arr)}
    print(label_json)

    session_list = sessions.copy()
    if type(sess) is not list:
        session = sess
        file_path = [session]
    else:
        session = "Allen_train_NN_test_zscored"
        file_path = sess

    signal_type = 'lfp' if 'lfp' in pickle_path else 'raw'
    freq_bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [13, 30], 'gamma': [30, 100]}

    train_sess_labels, val_sess_labels, test_sess_labels = [], [], []
    train_trial_chans, val_trial_chans, test_trial_chans = [], [], []

    print(f"Session list: {session_list}")
    test_features, test_labels, test_trials, test_chans = load_preprocessed_data(pickle_path, file_path, signal_type)
    features, labels, trials, chans = [], [], [], []
    for sessions in file_path:
        temp_features = test_features.get(sessions)
        temp_features = [f if f.shape[0] == 3749 else f[:3749] for f in temp_features]
        temp_features = np.array(temp_features)

        features.extend(temp_features)
        labels.extend(test_labels.get(sessions))
        trials.extend(test_trials.get(sessions))
        chans.extend(test_chans.get(sessions))

        temp_test_labels = test_labels.get(sessions)
        temp_test_labels = [f"{sessions}_{label_json[str(label)]}" for label in temp_test_labels]
        test_sess_labels.extend(temp_test_labels)

        temp_trials = test_trials.get(sessions)
        temp_test_chans = test_chans.get(sessions)
        temp_test_chans = [f"{sessions}_{str(temp_trials[i])}_{str(temp_test_chans[i])}" for i in
                           range(len(temp_trials))]
        test_trial_chans.extend(temp_test_chans)

    features = np.array(features)
    labels = np.array(labels)
    trials = np.array(trials)
    test_sess_labels = np.array(test_sess_labels)
    test_trial_chans = np.array(test_trial_chans)

    trial_length = 60
    train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)
    train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=0.25, random_state=42)
    test_idx = [idx for idx, val in enumerate(trials) if val in test_tr_idx]

    X_train, y_train, X_val, y_val = [], [], [], []
    X_test, y_test = features[test_idx], labels[test_idx]
    test_sess_labels = test_sess_labels[test_idx]
    test_trial_chans = test_trial_chans[test_idx]
    if type(sess) is not list:
        session_list.remove(session)
        random.seed(42)
        train_session_list = random.sample(session_list, int(len(session_list) * 0.8))
        val_session_list = [ses for ses in session_list if ses not in train_session_list]
        print("Training on ", train_session_list)
        print("Validate on ", val_session_list)
        print("Testing on ", session)
    else:
        random.seed(42)
        train_session_list = random.sample(session_list, int(len(session_list) * 0.8))
        val_session_list = [ses for ses in session_list if ses not in train_session_list]
        print("Training on ", train_session_list)
        print("Validate on ", val_session_list)

    all_features, all_labels, all_trials, all_chans = load_preprocessed_data(pickle_path, session_list, signal_type)
    for sess in train_session_list:
        features = all_features.get(sess)
        features = [f if f.shape[0] == 3749 else f[:3749] for f in features]
        features = np.array(features)
        labels = all_labels.get(sess)
        trial_idx = all_trials.get(sess)
        chans = all_chans.get(sess)

        idx_train = [idx for idx, val in enumerate(trial_idx) if val in train_tr_idx]

        temp_train_labels = labels[idx_train]
        temp_train_labels = [f"{sess}_{label_json[str(label)]}" for label in temp_train_labels]
        train_sess_labels.extend(temp_train_labels)

        temp_train_trials = trial_idx[idx_train]
        temp_train_chans = chans[idx_train]
        temp_train_chans = [f"{sess}_{str(temp_train_trials[i])}_{str(temp_train_chans[i])}"
                            for i in range(len(temp_train_trials))]
        train_trial_chans.extend(temp_train_chans)

        X_train.extend(features[idx_train])
        y_train.extend(labels[idx_train])

    for sess in val_session_list:
        features = all_features.get(sess)
        features = [f if f.shape[0] == 3749 else f[:3749] for f in features]
        features = np.array(features)
        labels = all_labels.get(sess)
        trial_idx = all_trials.get(sess)
        chans = all_chans.get(sess)

        idx_val = [idx for idx, val in enumerate(trial_idx) if val in val_tr_idx]

        temp_val_labels = labels[idx_val]
        temp_val_labels = [f"{sess}_{label_json[str(label)]}" for label in temp_val_labels]
        val_sess_labels.extend(temp_val_labels)

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
    train_sess_labels = np.array(train_sess_labels)
    val_sess_labels = np.array(val_sess_labels)
    train_trial_chans = np.array(train_trial_chans)
    val_trial_chans = np.array(val_trial_chans)

    y_train = [str(label) for label in y_train]
    y_val = [str(label) for label in y_val]
    y_test = [str(label) for label in y_test]

    assert len(y_train) == len(train_sess_labels), "Length of y_train and train_sess_labels are not equal"
    assert len(y_val) == len(val_sess_labels), "Length of y_val and val_sess_labels are not equal"
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

    # percent_to_stratify = 0.1
    # X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1-percent_to_stratify, stratify=y_train)
    # X_val, _, y_val, _ = train_test_split(X_val, y_val, test_size=1-percent_to_stratify, stratify=y_val)
    # X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=1-percent_to_stratify, stratify=y_test)


    id2label = {str(i): acr for i, acr in enumerate(acronyms_arr)}
    label2id = {acr: str(i) for i, acr in enumerate(acronyms_arr)}

    # X_train = np.array(X_train[:100])
    # y_train = y_train[:100]
    #
    # X_val = np.array(X_val[:100])
    # y_val = y_val[:100]
    #
    # X_test = np.array(X_test[:100])
    # y_test = y_test[:100]
    custom_features = Features({
        'label': ClassLabel(num_classes=5, names=['CA1', 'CA2', 'CA3', 'DG', 'VIS']),
        'input_values': Sequence(feature=Value(dtype='float'), length=-1)
    })

    print(f"label2id: {label2id}")
    print(f"id2label: {id2label}")
    print(f"Features: {custom_features}")
    print(f"Unique labels in dataset: {np.unique(y_train + y_val + y_test)}")
    print(f"Expected labels in id2label: {list(id2label.keys())}")

    print("Upsampling the data...")
    fq = freq_bands[freq_band]
    print(fq)
    X_train_upsampled = preprocess_data(X_train, sampling_rate, [fq[0], fq[1]])
    X_val_upsampled = preprocess_data(X_val, sampling_rate, [fq[0], fq[1]])
    X_test_upsampled = preprocess_data(X_test, sampling_rate, [fq[0], fq[1]])

    temp1 = X_train[0]
    temp2 = X_train_upsampled[0]
    plot(temp1, temp2, f"{freq_band}.png")
    if load_data and os.path.exists(f"{data_loading_path}/{session}_{freq_band}_dataset_dict.pickle"):
        print(f"Loading from {data_loading_path}/{session}_{freq_band}_dataset_dict.pickle")
        with open(f"{data_loading_path}/{session}_{freq_band}_dataset_dict.pickle", 'rb') as f:
            dataset_dict = pickle.load(f)
    else:
        print("Creating dataset dictionary...")
        dataset_dict = create_dataset_dict(X_train_upsampled, y_train, X_val_upsampled, y_val
                                            , X_test_upsampled, y_test, custom_features)
        if load_data:
            with open(f"{output_path}/{session}_{freq_band}_dataset_dict.pickle", 'wb') as f:
                pickle.dump(dataset_dict, f)
            return None
            # exit(0)

    def generate_random_string(length):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    random_string = generate_random_string(6)
    accuracy = evaluate.load("accuracy", experiment_id=random_string)

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

    num_labels = len(id2label)

    print("Training the model...")
    if args.data == 'ibl':
        w2v2_config = {"vocab_size": 32, "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
                   "intermediate_size": 3072, "hidden_act": "gelu", "hidden_dropout": 0.1,
                   "attention_dropout": 0.1, "final_dropout": 0.1, "initializer_range": 0.02,
                   "layer_norm_eps": 1e-12, "feat_extract_norm": "group", "feat_proj_dropout": 0.0,
                   "feat_extract_activation": "gelu", "feat_quantizer_dropout": 0.0,
                   "conv_dim": (512, 512, 512, 512, 512, 512, 512),
                   "conv_stride": (5, 2, 2, 2, 2, 2, 2), "conv_kernel": (10, 3, 3, 3, 3, 3, 3), "conv_bias": False,
                   "num_conv_pos_embeddings": 128, "num_conv_pos_embeddings_groups": 16, "do_stable_layer_norm": False,
                   "apply_spec_augment": True, "mask_time_prob": 0.05, "mask_time_length": 10, "mask_feature_prob": 0.0,
                   "mask_feature_length": 10, "num_codevectors_per_group": 320, "num_codevector_groups": 2,
                   "contrastive_logits_temperature": 0.1, "num_negatives": 100, "codevector_dim": 256,
                   "proj_codevector_dim": 256, "diversity_loss_weight": 0.1, "ctc_loss_reduction": "sum",
                   "ctc_zero_infinity": False, "use_weighted_layer_sum": False, "classifier_proj_size": 256}
    else:
        w2v2_config = {"vocab_size": 32, "hidden_size": 1000, "num_hidden_layers": 12, "num_attention_heads": 20,
                       "intermediate_size": 3072, "hidden_act": "gelu", "hidden_dropout": 0.1,
                       "attention_dropout": 0.1, "final_dropout": 0.1, "initializer_range": 0.02,
                       "layer_norm_eps": 1e-12, "feat_extract_norm": "group", "feat_proj_dropout": 0.0,
                       "feat_extract_activation": "gelu", "feat_quantizer_dropout": 0.0,
                       "conv_dim": (512, 512, 512, 512, 512, 512, 512),
                       "conv_stride": (5, 2, 2, 2, 2, 2, 2), "conv_kernel": (10, 3, 3, 3, 3, 3, 3), "conv_bias": False,
                       "num_conv_pos_embeddings": 128, "num_conv_pos_embeddings_groups": 16, "do_stable_layer_norm": False,
                       "apply_spec_augment": True, "mask_time_prob": 0.05, "mask_time_length": 10, "mask_feature_prob": 0.0,
                       "mask_feature_length": 10, "num_codevectors_per_group": 320, "num_codevector_groups": 2,
                       "contrastive_logits_temperature": 0.1, "num_negatives": 100, "codevector_dim": 256,
                       "proj_codevector_dim": 256, "diversity_loss_weight": 0.1, "ctc_loss_reduction": "sum",
                       "ctc_zero_infinity": False, "use_weighted_layer_sum": False, "classifier_proj_size": 300}

    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label, config=w2v2_config
    )

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model is on device: {model.device}")

    training_args = TrainingArguments(
        output_dir=f"{output_path}/{session}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=12,
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
        eval_dataset=dataset_dict["validation"].with_format("torch"),
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
    )

    best_ckpt_path = None
    trainer.train()
    best_ckpt_path = trainer.state.best_model_checkpoint

    projector_hook_handle = model.projector.register_forward_hook(projector_hook)
    classifier_hook_handle = model.classifier.register_forward_hook(classifier_hook)

    train_embeddings, val_embeddings, test_embeddings = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    # Train embeddings
    current_index = 0
    batch_size = 30

    # Flatten the dataset for easier sequential access
    full_data = X_train_upsampled
    full_labels = y_train

    # Continue until the entire dataset is processed
    while current_index < len(full_data):
        print(f"Train {current_index}/{len(full_data)}")
        # Take the next batch of 30 samples
        end_index = min(current_index + batch_size, len(full_data))
        batch_data = full_data[current_index:end_index]
        batch_labels = full_labels[current_index:end_index]

        # Update the current index
        current_index = end_index

        # Convert batch to required format
        batch_dict = {"label": batch_labels, "input_values": batch_data.tolist()}
        batch_dataset = Dataset.from_dict(batch_dict, features=custom_features)

        # Get predictions
        small_predictions = trainer.predict(batch_dataset.with_format("torch"))

        # Get embeddings for the current batch
        temp_embeddings = classifier_input_first.cpu().detach().numpy()

        # Save embeddings and labels
        train_embeddings.extend(temp_embeddings)
        train_labels.extend(batch_labels)

    # Validation embeddings
    current_index = 0
    batch_size = 30

    # Flatten the dataset for easier sequential access
    full_data = X_val_upsampled
    full_labels = y_val

    # Continue until the entire dataset is processed
    while current_index < len(full_data):
        print(f"Val {current_index}/{len(full_data)}")
        # Take the next batch of 30 samples
        end_index = min(current_index + batch_size, len(full_data))
        batch_data = full_data[current_index:end_index]
        batch_labels = full_labels[current_index:end_index]

        # Update the current index
        current_index = end_index

        # Convert batch to required format
        batch_dict = {"label": batch_labels, "input_values": batch_data.tolist()}
        batch_dataset = Dataset.from_dict(batch_dict, features=custom_features)

        # Get predictions
        small_predictions = trainer.predict(batch_dataset.with_format("torch"))

        # Get embeddings for the current batch
        temp_embeddings = classifier_input_first.cpu().detach().numpy()

        # Save embeddings and labels
        val_embeddings.extend(temp_embeddings)
        val_labels.extend(batch_labels)

    # Test embeddings
    current_index = 0
    batch_size = 30

    # Flatten the dataset for easier sequential access
    full_data = X_test_upsampled
    full_labels = y_test

    # Continue until the entire dataset is processed
    while current_index < len(full_data):
        print(f"Test {current_index}/{len(full_data)}")
        # Take the next batch of 30 samples
        end_index = min(current_index + batch_size, len(full_data))
        batch_data = full_data[current_index:end_index]
        batch_labels = full_labels[current_index:end_index]

        # Update the current index
        current_index = end_index

        # Convert batch to required format
        batch_dict = {"label": batch_labels, "input_values": batch_data.tolist()}
        batch_dataset = Dataset.from_dict(batch_dict, features=custom_features)

        # Get predictions
        small_predictions = trainer.predict(batch_dataset.with_format("torch"))

        # Get embeddings for the current batch
        temp_embeddings = classifier_input_first.cpu().detach().numpy()

        # Save embeddings and labels
        test_embeddings.extend(temp_embeddings)
        test_labels.extend(batch_labels)

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

    file_path = os.path.join(output_path, f'{session}_{freq_band}_results.pickle')
    file_obj = {
        'train_logits': train_logits,  # Predicted logits for training set
        'train_labels': train_labels,  # True labels for training set
        'train_acc': train_acc,  # Accuracy for training set
        'train_embeddings': train_embeddings,  # Projector layer Embeddings for training set
        'train_trial_chans': train_trial_chans,  # Trial and channel IDs for training set ("<Session>_<Trial>_<Chan>")
        'train_sess_labels': train_sess_labels,  # Session labels for training set ("<Session>_<Label>")
        'val_logits': val_logits,  # Predicted logits for validation set
        'val_labels': val_labels,  # True labels for validation set
        'val_acc': val_acc,  # Accuracy for validation set
        'val_embeddings': val_embeddings,  # Projector layer Embeddings for validation set
        'val_trial_chans': val_trial_chans,  # Trial and channel IDs for validation set ("<Session>_<Trial>_<Chan>")
        'val_sess_labels': val_sess_labels,  # Session labels for validation set ("<Session>_<Label>")
        'test_logits': test_logits,  # Predicted logits for test set
        'test_labels': test_labels,  # True labels for test set
        'test_acc': test_acc,  # Accuracy for test set
        'test_embeddings': test_embeddings,  # Projector layer Embeddings for test set
        'test_trial_chans': test_trial_chans,  # Trial and channel IDs for test set ("<Session>_<Trial>_<Chan>")
        'test_sess_labels': test_sess_labels,  # Session labels for test set ("<Session>_<Label>")
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
        print(f"Session {session} results saved to {file_path}")
        print(file_obj['train_acc'], file_obj['val_acc'], file_obj['test_acc'])

    visualizer = PCAVisualizer(label_json, output_path=output_path)
    try:
        visualizer.create_pca(train_embeddings, train_labels, 2, session, "train")
        visualizer.create_pca(train_embeddings, train_labels, 3, session, "train")
        visualizer.create_pca(val_embeddings, val_labels, 2, session, "val")
        visualizer.create_pca(val_embeddings, val_labels, 3, session, "val")
        visualizer.create_pca(test_embeddings, test_labels, 2, session, "test")
        visualizer.create_pca(test_embeddings, test_labels, 3, session, "test")

        datasets = {
            "train": {"embeddings": train_embeddings, "labels": train_labels},
            "val": {"embeddings": val_embeddings, "labels": val_labels},
            "test": {"embeddings": test_embeddings, "labels": test_labels}
        }
        visualizer.create_combined_pca(datasets, 2, session, "combined")
        visualizer.create_combined_pca(datasets, 3, session, "combined")
    except:
        print(f"Possible size mismatch: "
            f"Train({len(train_embeddings)}, {len(train_labels)}), "
            f"Val({len(val_embeddings)}, {len(val_labels)}), "
            f"Test({len(test_embeddings)}, {len(test_labels)})")
    

    # Deregister hooks
    projector_hook_handle.remove()
    classifier_hook_handle.remove()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if data == "Allen":
        sessions_list = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
        pickle_path = "spectrogram/Allen"
        hc_acronyms = ['CA1', 'CA2', 'CA3', 'DG', 'VIS']
    elif data == 'ibl':
        sessions_list = ['0802ced5-33a3-405e-8336-b65ebc5cb07c_probe00',
                         '0802ced5-33a3-405e-8336-b65ebc5cb07c_probe01',
                         '0a018f12-ee06-4b11-97aa-bbbff5448e9f_probe00',
                         '3638d102-e8b6-4230-8742-e548cd87a949_probe01',
                         '5dcee0eb-b34d-4652-acc3-d10afc6eae68_probe00',
                         'd2832a38-27f6-452d-91d6-af72d794136c_probe00',
                         '54238fd6-d2d0-4408-b1a9-d19d24fd29ce_probe00']
        pickle_path = f'/vast/th3129/data/ibl_new/spectrogram_preprocessed'
        hc_acronyms = ['CA1', 'CA2', 'CA3', 'DG', 'VIS']
    elif data == "Neuronexus":
        sessions_list = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02']
        pickle_path = f'/scratch/th3129/region_decoding/data/Neuronexus/lfp'
        hc_acronyms = ['Cortex', 'CA1', 'CA2', 'CA3', 'DG']
    elif data == "All":
        sessions_list = ['719161530', '794812542', '778998620', '798911424', '771160300', '768515987', '771990200']
        test_sess = ['AD_HF01_1', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02', 'AD_HF02_2']
        pickle_path = "spectrogram/Allen"

    freq_bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [13, 30], 'gamma': [30, 100]}
    # freq_bands = {'delta': [0.5, 4], 'theta': [4, 8]}
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    to_skip = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_02']

    # Hooks
    projector_output = None
    classifier_input_first = None
    classifier_input_whole = None
    classifier_output = None

    if data != "All":
        for key in freq_bands.keys():
            for i in range(len(sessions_list)):
                if sessions_list[i] in to_skip:
                    print(f"Skipping {sessions_list[i]}...")
                    continue
                run_wave2vec2(sessions_list, sessions_list[i], int(sampling_rate), hc_acronyms, key)
    else:
        run_wave2vec2(sessions_list, test_sess, int(sampling_rate))