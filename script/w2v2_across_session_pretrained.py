import argparse
import os
import pickle
import random

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.signal import resample
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence, Value
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer, pipeline
from blind_localization.data.PCAviz import PCAVisualizer


# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='wave2vec2')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen or ibl', default='ibl')
    parser.add_argument('--trial_length', type=int, default=60, help='trial_length')
    parser.add_argument('--data_type', type=str, help='Data type to use', default='spectrogram_preprocessed')
    parser.add_argument('--sampling_rate', type=str, help='Sampling rate of the data', default='1250')
    return parser.parse_args()


args = arg_parser()
data, trial_length, data_type, sampling_rate = args.data, args.trial_length, args.data_type, args.sampling_rate
print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}, Sampling Rate: {sampling_rate}")

output_path = f"../results/{data}/{data_type}/wave2vec2/across_session"
if not os.path.exists(output_path):
    os.makedirs(output_path)


def load_preprocessed_data(pickle_path, file_path):
    features, labels, trials = {}, {}, {}
    for session in file_path:
        data = pickle.load(open(f"{pickle_path}/{session}_raw.pickle", 'rb'))
        X, y, trial_idx = zip(*[(d[0], d[1], d[2]) for d in data])
        X = [x if x.shape[0] == 3749 else x[:3749] for x in X]
        features[session] = np.array(X)
        labels[session] = np.array(y, dtype=int)
        trials[session] = np.array(trial_idx)

        non_zero_indices = [i for i, x in enumerate(features[session]) if not np.all(x == 0)]

        features[session] = features[session][non_zero_indices]
        labels[session] = labels[session][non_zero_indices]
        trials[session] = trials[session][non_zero_indices]

        # Sanity check
        assert len(features[session]) == len(labels[session]) == len(trials[session]), \
            f"Inconsistent data sizes for session {session}"

    return features, labels, trials


def preprocess_data(signals, signal_sampling_rate):
    target_sampling_rate = 16000
    upsampled_signals = []

    for signal in signals:
        # Calculate the number of samples for the target sampling rate
        num_target_samples = int(len(signal) * target_sampling_rate / signal_sampling_rate)
        # Resample the signal
        upsampled_signal = resample(signal, num_target_samples)
        upsampled_signals.append(upsampled_signal)

    return upsampled_signals


def create_dataset_dict(X_train, y_train, X_val, y_val, X_test, y_test, custom_features):
    # Create individual datasets
    print(len(X_train), len(y_train), len(X_val), len(y_val), len(X_test), len(y_test))
    train_dict = {"label": y_train, "input_values": X_train if type(X_train) is not list else X_train}
    # print(X_train.shape)
    train_dataset = Dataset.from_dict(train_dict, features=custom_features)

    val_dict = {"label": y_val, "input_values": X_val if type(X_val) is not list else X_val}
    # print(X_val.shape)
    val_dataset = Dataset.from_dict(val_dict, features=custom_features)

    test_dict = {"label": y_test, "input_values": X_test if type(X_test) is not list else X_test}
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


def run_wave2vec2(sessions, sess, sampling_rate):
    session_list = sessions
    session = sess
    print(f"Session list: {session_list}")
    file_path = [session]
    features, labels, trials = load_preprocessed_data(pickle_path, file_path)
    features = features.get(file_path[0])
    labels = labels.get(file_path[0])
    trials = trials.get(file_path[0])
    
    trial_length = 60
    train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)
    train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=0.25, random_state=42)
    test_idx = [idx for idx, val in enumerate(trials) if val not in train_tr_idx]

    X_train, y_train, X_val, y_val = [], [], [], []
    X_test, y_test = features[test_idx], labels[test_idx]
    session_list.remove(session)
    session_list = random.sample(session_list, 4)

    all_features, all_labels, all_trials = load_preprocessed_data(pickle_path, session_list)
    for sess in session_list:
        features = all_features.get(sess)
        features = [f if f.shape[0] == 3749 else f[:3749] for f in features]
        features = np.array(features)
        labels = all_labels.get(sess)
        trial_idx = all_trials.get(sess)

        idx_train = [idx for idx, val in enumerate(trial_idx) if val in train_tr_idx]
        idx_val = [idx for idx, val in enumerate(trial_idx) if val in val_tr_idx]

        X_train.extend(features[idx_train])
        y_train.extend(labels[idx_train])

        X_val.extend(features[idx_val])
        y_val.extend(labels[idx_val])
    
    # X_train = np.array(X_train)
    # X_val = np.array(X_val)
    # X_test = np.array(X_test)

    y_train = [str(label) for label in y_train]
    y_val = [str(label) for label in y_val]
    y_test = [str(label) for label in y_test]

    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
    acronyms_arr = np.array(sorted(list(hc_acronyms)))
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
    X_train_upsampled = preprocess_data(X_train, sampling_rate)
    X_val_upsampled = preprocess_data(X_val, sampling_rate)
    X_test_upsampled = preprocess_data(X_test, sampling_rate)

    # temp1 = X_train[0]
    # temp2 = X_train_upsampled[0]
    # plot(temp1, temp2)

    print("Creating dataset dictionary...")
    dataset_dict = create_dataset_dict(X_train_upsampled, y_train, X_val_upsampled, y_val
                                       , X_test_upsampled, y_test, custom_features)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

    num_labels = len(id2label)

    print("Loading the model...")
    model_dir = "/scratch/mkp6112/LFP/region_decoding/results/Allen/spectrogram/wave2vec2/across_session/794812542/checkpoint-3660/"
    model = AutoModelForAudioClassification.from_pretrained(
        model_dir, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir=f"{output_path}/{session}",
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"].with_format("torch"),
        eval_dataset=dataset_dict["validation"].with_format("torch"),
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
    )

    # trainer.train()

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

    visualizer = PCAVisualizer(output_path=output_path)

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
    visualizer.create_combined_pca(datasets, 2, session)
    visualizer.create_combined_pca(datasets, 3, session)

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

    file_path = os.path.join(output_path, f'{session}_results.pickle')
    with open(file_path, 'wb') as f:
        pickle.dump({
            'train_logits': train_logits,
            'train_labels': train_labels,
            'train_acc': train_acc,
            'val_logits': val_logits,
            'val_labels': val_labels,
            'val_acc': val_acc,
            'test_logits': test_logits,
            'test_labels': test_labels,
            'test_acc': test_acc
        }, f)

    # Deregister hooks
    projector_hook_handle.remove()
    classifier_hook_handle.remove()


if __name__ == "__main__":
    if data == "Allen":
        sessions_list = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
        pickle_path = "spectrogram/Allen"
    elif data == 'ibl':
        sessions_list = ['15763234-d21e-491f-a01b-1238eb96d389', '1a507308-c63a-4e02-8f32-3239a07dc578',
                     '4a45c8ba-db6f-4f11-9403-56e06a33dfa4', '56956777-dca5-468c-87cb-78150432cc57',
                     '5b49aca6-a6f4-4075-931a-617ad64c219c', '73918ae1-e4fd-4c18-b132-00cb555b1ad2',
                     'b39752db-abdb-47ab-ae78-e8608bbf50ed']
        pickle_path = f'/scratch/cl7201/shared/{data}/{data_type}'
        # pickle_path = f'spectrogram/ibl'
        hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
    elif data == "Neuronexus":
        sessions_list = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]
        pickle_path = f'/scratch/th3129/region_decoding/data/Neuronexus/lfp'
        hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Cortex'}

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # Hooks
    projector_output = None
    classifier_input_first = None
    classifier_input_whole = None
    classifier_output = None

    for i in range(len(sessions_list)):
        run_wave2vec2(sessions_list, sessions_list[i], int(sampling_rate))