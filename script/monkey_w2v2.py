import argparse
import os
import librosa
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
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence, Value
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer, pipeline
from blind_localization.data.PCAviz import PCAVisualizer


# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='wave2vec2')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen or ibl', default='Monkey')
    parser.add_argument('--trial_length', type=int, default=60, help='trial_length')
    parser.add_argument('--data_type', type=str, help='Data type to use', default='spectrogram')
    parser.add_argument('--sampling_rate', type=str, help='Sampling rate of the data', default='2500')
    return parser.parse_args()


args = arg_parser()
train_session_list, val_session_list, w2v2_session_name, best_ckpt_path_parent = None, None, None, None
data, trial_length, data_type, sampling_rate = args.data, args.trial_length, args.data_type, args.sampling_rate
print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}, Sampling Rate: {sampling_rate}")
print(torch.cuda.is_available())

output_path = f"../results/{data}/{data_type}/wave2vec2/across_session"
if not os.path.exists(output_path):
    os.makedirs(output_path)


def load_preprocessed_data(pickle_path, file_path):
    features, labels, trials, chans = {}, {}, {}, {}
    for session in file_path:
        print(f"Loading {pickle_path}/{session}.pickle")
        data = pickle.load(open(f"{pickle_path}/{session}.pickle", 'rb'))
        X, y, trial_idx, chan_id = zip(*[(d[0], d[1], d[2], d[3]) for d in data])
        y = [label.replace("imec", "") for label in y]
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

    for signal in tqdm(signals):
        num_target_samples = int(len(signal) * target_sampling_rate / signal_sampling_rate)
        upsampled_signal = resample(signal, num_target_samples)
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


def plot(temp1, temp2):
    plt.figure(figsize=(10, 6))
    plt.plot(temp1, label='Original')
    plt.plot(temp2, label='Upsampled')
    plt.legend()
    plt.show()


def projector_hook(module, input, output):
    global projector_output
    projector_output = output


def classifier_hook(module, input, output):
    global classifier_input_first, classifier_input_whole, classifier_output
    classifier_input_first = input[0]
    classifier_input_whole = input
    classifier_output = output


def run_wave2vec2(sessions, sess, sampling_rate, train_session_list=None, val_session_list=None):
    label_json = {"0": "Basal_Ganglia", "1": "Suppl_Motor_Area", "2": "Primary_Motor_Cortex"}
    session_list = sessions.copy()
    custom_trial = None
    if type(sess) is not list:
        session = sess
        file_path = [session]
    else:
        try:
            custom_trial = int(w2v2_session_name.replace("Allen_test_", ""))
        except ValueError:
            custom_trial = None
        session = w2v2_session_name
        file_path = sess

    train_sess_labels, val_sess_labels, test_sess_labels = [], [], []
    train_trial_chans, val_trial_chans, test_trial_chans = [], [], []

    print(f"Session list: {session_list}")
    test_features, test_labels, test_trials, test_chans = load_preprocessed_data(pickle_path, file_path)
    features, labels, trials, chans = [], [], [], []
    for sessions in file_path:
        temp_features = test_features.get(sessions)
        temp_features = np.array(temp_features)

        features.extend(temp_features)
        labels.extend(test_labels.get(sessions))
        trials.extend(test_trials.get(sessions))
        chans.extend(test_chans.get(sessions))

        temp_test_labels = test_labels.get(sessions)
        temp_test_labels = [f"{sessions}_{str(label)}" for label in temp_test_labels]
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

    trial_length = custom_trial if custom_trial else 60
    print(f"Using custom trial: {trial_length}")
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
    else:
        random.seed(42)
        print("Training on ", train_session_list)
        print("Validate on ", val_session_list)

    all_features, all_labels, all_trials, all_chans = load_preprocessed_data(pickle_path, session_list)
    print(list(all_features.keys()))
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
        print(sess)
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
    print(train_label_counts, val_label_counts, test_label_counts)

    # percent_to_stratify = 0.1
    # X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1-percent_to_stratify, stratify=y_train)
    # X_val, _, y_val, _ = train_test_split(X_val, y_val, test_size=1-percent_to_stratify, stratify=y_val)
    # X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=1-percent_to_stratify, stratify=y_test)
    hc_acronyms = ['Basal_Ganglia', 'Suppl_Motor_Area', 'Primary_Motor_Cortex']
    acronyms_arr = np.array(list(hc_acronyms))
    id2label = {str(i): acr for i, acr in enumerate(acronyms_arr)}
    label2id = {acr: str(i) for i, acr in enumerate(acronyms_arr)}

    custom_features = Features({
        'label': ClassLabel(num_classes=3, names=['Basal_Ganglia', 'Suppl_Motor_Area', 'Primary_Motor_Cortex']),
        'input_values': Sequence(feature=Value(dtype='float'), length=-1)
    })

    print(f"label2id: {label2id}")
    print(f"id2label: {id2label}")
    print(f"Features: {custom_features}")
    print(f"Unique labels in dataset: {np.unique(y_train + y_val + y_test)}")
    print(f"Expected labels in id2label: {list(id2label.keys())}")

    print("Upsampling the data...")
    X_train_upsampled = preprocess_data(X_train, sampling_rate)
    X_val_upsampled = preprocess_data(X_val, sampling_rate)
    X_test_upsampled = preprocess_data(X_test, sampling_rate)

    # temp1 = X_train[0]
    # temp2 = X_train_upsampled[0]
    # plot(temp1, temp2)

    print("Creating dataset dictionary...")
    if os.path.exists(f"{output_path}/{session}_dataset_dict.pickle"):
        with open(f"{output_path}/{session}_dataset_dict.pickle", 'rb') as f:
            dataset_dict = pickle.load(f)
    else:
        dataset_dict = create_dataset_dict(X_train_upsampled, y_train, X_val_upsampled, y_val
                                           , X_test_upsampled, y_test, custom_features)
        with open(f"{output_path}/{session}_dataset_dict.pickle", 'wb') as f:
            pickle.dump(dataset_dict, f)
        return None

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
                       "num_conv_pos_embeddings": 128, "num_conv_pos_embeddings_groups": 16,
                       "do_stable_layer_norm": False,
                       "apply_spec_augment": True, "mask_time_prob": 0.05, "mask_time_length": 10,
                       "mask_feature_prob": 0.0,
                       "mask_feature_length": 10, "num_codevectors_per_group": 320, "num_codevector_groups": 2,
                       "contrastive_logits_temperature": 0.1, "num_negatives": 100, "codevector_dim": 256,
                       "proj_codevector_dim": 256, "diversity_loss_weight": 0.1, "ctc_loss_reduction": "sum",
                       "ctc_zero_infinity": False, "use_weighted_layer_sum": False, "classifier_proj_size": 256}
    else:
        w2v2_config = {"vocab_size": 32, "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
                       "intermediate_size": 3072, "hidden_act": "gelu", "hidden_dropout": 0.1,
                       "attention_dropout": 0.1, "final_dropout": 0.1, "initializer_range": 0.02,
                       "layer_norm_eps": 1e-12, "feat_extract_norm": "group", "feat_proj_dropout": 0.0,
                       "feat_extract_activation": "gelu", "feat_quantizer_dropout": 0.0,
                       "conv_dim": (512, 512, 512, 512, 512, 512, 512),
                       "conv_stride": (5, 2, 2, 2, 2, 2, 2), "conv_kernel": (10, 3, 3, 3, 3, 3, 3), "conv_bias": False,
                       "num_conv_pos_embeddings": 128, "num_conv_pos_embeddings_groups": 16,
                       "do_stable_layer_norm": False,
                       "apply_spec_augment": True, "mask_time_prob": 0.05, "mask_time_length": 10,
                       "mask_feature_prob": 0.0,
                       "mask_feature_length": 10, "num_codevectors_per_group": 320, "num_codevector_groups": 2,
                       "contrastive_logits_temperature": 0.1, "num_negatives": 100, "codevector_dim": 256,
                       "proj_codevector_dim": 256, "diversity_loss_weight": 0.1, "ctc_loss_reduction": "sum",
                       "ctc_zero_infinity": False, "use_weighted_layer_sum": False, "classifier_proj_size": 256}

    if args.data != 'All':
        model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label,
            config=w2v2_config
        )
    else:
        # model_dir = "../results/All/spectrogram/wave2vec2/across_session/mix/checkpoint-11340"
        # model = AutoModelForAudioClassification.from_pretrained(
        #  model_dir, num_labels=num_labels, label2id=label2id, id2label=id2label
        # )
        model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label,
            config=w2v2_config
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
        per_device_train_batch_size=50,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=50,
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

    if best_ckpt_path_parent is not None or os.path.exists(os.path.join(output_path, f'{session}_results.pickle')):
        print(best_ckpt_path_parent)
        if best_ckpt_path_parent:
            best_ckpt_path = best_ckpt_path_parent
        else:
            with open(os.path.join(output_path, f'{session}_results.pickle'), 'rb') as f:
                existing_data = pickle.load(f)
                best_ckpt_path = existing_data['best_ckpt_path']
        model = AutoModelForAudioClassification.from_pretrained(
            best_ckpt_path, num_labels=num_labels, label2id=label2id, id2label=id2label
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"].with_format("torch"),
            eval_dataset=dataset_dict["validation"].with_format("torch"),
            processing_class=feature_extractor,
            compute_metrics=compute_metrics,
        )
        model.to(device)
        print(f"Model loaded from {best_ckpt_path}, on device {device}")
    else:
        best_ckpt_path = None
        if args.data != 'All':
            trainer.train()
            best_ckpt_path = trainer.state.best_model_checkpoint
        else:
            trainer.train()
            best_ckpt_path = trainer.state.best_model_checkpoint
        print(f"Best checkpoint path after training: {best_ckpt_path}")

    train_pred = trainer.predict(dataset_dict["train"].with_format("torch"))
    train_acc = train_pred[2]['test_accuracy']

    val_pred = trainer.predict(dataset_dict["validation"].with_format("torch"))
    val_acc = val_pred[2]['test_accuracy']

    test_pred = trainer.predict(dataset_dict["test"].with_format("torch"))
    test_acc = test_pred[2]['test_accuracy']
    print(f"Train_acc: {train_acc}, Val acc: {val_acc}, Test acc: {test_acc}")
    print(f"Train chance accuracy: {train_chance_accuracy}, Validation chance accuracy: {val_chance_accuracy}, "
          f"Test chance accuracy: {test_chance_accuracy}")

    projector_hook_handle = model.projector.register_forward_hook(projector_hook)
    classifier_hook_handle = model.classifier.register_forward_hook(classifier_hook)

    train_embeddings, val_embeddings, test_embeddings = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    # Train embeddings
    current_index = 0
    batch_size = 50

    # Flatten the dataset for easier sequential access
    full_data = X_train_upsampled
    full_labels = y_train
    print(len(X_train_upsampled), len(y_train))
    print(len(X_val_upsampled), len(y_val))
    print(len(X_test_upsampled), len(y_test))

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
    batch_size = 50

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
    batch_size = 50

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

    if os.path.exists(os.path.join(output_path, f'{session}_results.pickle')):
        with open(os.path.join(output_path, f'{session}_results.pickle'), 'rb') as f:
            existing_data = pickle.load(f)
        if existing_data['val_acc'] > val_acc:
            print(f"Skipping session {session} as it has lower accuracy than existing data.")
            print(f"Skipped session's accs; train_acc: {train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")
            print(f"Skipped session's config: {w2v2_config}")
            return None

    visualizer = PCAVisualizer(label_json, output_path=output_path)

    print(f"Creating PCA visualizations for session {session}...")
    visualizer.create_pca(train_embeddings, train_labels, 2, session, "train")
    visualizer.create_pca(train_embeddings, train_labels, 3, session, "train")
    visualizer.create_pca(val_embeddings, val_labels, 2, session, "val")
    visualizer.create_pca(val_embeddings, val_labels, 3, session, "val")
    visualizer.create_pca(test_embeddings, test_labels, 2, session, "test")
    visualizer.create_pca(test_embeddings, test_labels, 3, session, "test")

    print(f"Creating combined PCA visualizations for session {session}...")
    datasets = {
        "train": {"embeddings": train_embeddings, "labels": train_labels},
        "val": {"embeddings": val_embeddings, "labels": val_labels},
        "test": {"embeddings": test_embeddings, "labels": test_labels}
    }
    visualizer.create_combined_pca(datasets, 2, session)
    visualizer.create_combined_pca(datasets, 3, session)

    print(f"Creating session-wise PCA visualizations for session {session}...")
    datasets = {
        "train": {"embeddings": train_embeddings, "labels": train_sess_labels},
        "val": {"embeddings": val_embeddings, "labels": val_sess_labels},
        "test": {"embeddings": test_embeddings, "labels": test_sess_labels}
    }
    visualizer.create_combined_pca(datasets, 2, session, "session_wise")
    visualizer.create_combined_pca(datasets, 3, session, "session_wise")

    file_path = os.path.join(output_path, f'{session}_results.pickle')
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
        print(f"Train accuracy: {train_acc}, Validation accuracy: {val_acc}, Test accuracy: {test_acc}")
        print(f"Train label counts: {train_label_counts}, Validation label counts: {val_label_counts}, "
              f"Test label counts: {test_label_counts}")
        print(f"Train chance accuracy: {train_chance_accuracy}, Validation chance accuracy: {val_chance_accuracy}, "
              f"Test chance accuracy: {test_chance_accuracy}")
        print(f"w2v2_config: {w2v2_config}, best_ckpt_path: {best_ckpt_path}")

    # Deregister hooks
    projector_hook_handle.remove()
    classifier_hook_handle.remove()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    conf = {
        "221007": [
            ['221104', '221216'], ['221216'], ['221007'],
            ['../results/Monkey/spectrogram/wave2vec2/across_session/221007/checkpoint-12672']
        ],

        "221104": [
            ['221007', '221216'], ['221007'], ['221104'],
            [None]
        ],

        "221216": [
            ['221104', '221007'], ['221104'], ['221216'],
            [None]
        ]
    }

    sessions_list = None
    test_sess = None
    val_list = None
    pickle_path = "spectrogram/Allen"

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    to_skip = []

    # Hooks
    projector_output = None
    classifier_input_first = None
    classifier_input_whole = None
    classifier_output = None

    for keys in conf.keys():
        w2v2_session_name = keys
        best_ckpt_path_parent = conf[keys][3][0]
        train_session_list = conf[keys][0]
        val_session_list = conf[keys][1]
        test_sess = conf[keys][2]
        print(f"Running {keys}...")
        whole_list = []
        for i in range(len(train_session_list)):
            whole_list.append(train_session_list[i])
        for i in range(len(val_session_list)):
            whole_list.append(val_session_list[i])
        run_wave2vec2(whole_list, test_sess, int(sampling_rate), train_session_list, val_session_list)