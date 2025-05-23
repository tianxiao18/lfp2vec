"""
visualization package

"""
import json
import os
import wandb
import pickle
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from script.utils import save_if_better
from torch import optim
from functools import partial
from blind_localization.data.datasets import RawDataset
from blind_localization.models.contrastive_pipeline import *
from blind_localization.models.contrastive import *
from blind_localization.models.decoder import *
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch import nn
import torch
import numpy as np
import mplcursors
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA    
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_preprocessed_data(pickle_path, file_path):
    """
    Load raw data from specified pickle files and preprocess it.

    Args:
        pickle_path (str): Path to the directory containing the pickle files.
        file_path (list): List of session file names.

    Returns:
        tuple: Dictionaries containing features, labels, trials, and channels for each session.
    """
    features, labels, trials, channels = {}, {}, {}, {}
    for session in file_path:
        data = pickle.load(open(f"{pickle_path}/{session}_raw.pickle", 'rb'))
        X, y, trial_idx, chan_idx = zip(*[(d[0], d[1], d[2], d[3]) for d in data])
        features[session] = np.array(X)
        labels[session] = np.array(y, dtype=int)
        trials[session] = np.array(trial_idx)
        channels[session] = np.array(chan_idx)

        non_zero_indices = [i for i, x in enumerate(features[session]) if not np.all(x == 0)]

        features[session] = features[session][non_zero_indices]
        labels[session] = labels[session][non_zero_indices]
        trials[session] = trials[session][non_zero_indices]
        channels[session] = channels[session][non_zero_indices]

        # Sanity check
        assert len(features[session]) == len(labels[session]) == len(trials[session]), \
            f"Inconsistent data sizes for session {session}"

    return features, labels, trials, channels

def build_single_session_full_loader(channel_features_all, channel_labels_all, channel_trials_all, channel_channels_all,
                                    config, Dataset, sessions, session_idx=0):
    """
    Build a data loader for a single session to load the full dataset.

    Args:
        channel_features_all (dict): Dictionary of features for all sessions.
        channel_labels_all (dict): Dictionary of labels for all sessions.
        channel_trials_all (dict): Dictionary of trial indices for all sessions.
        channel_channels_all (dict): Dictionary of channel indices for all sessions.
        config (dict): Configuration dictionary containing time_bins, library, and batch_size.
        Dataset (Dataset): Custom dataset class for loading data.
        sessions (list): List of session names.
        session_idx (int, optional): Index of the session to be loaded. Defaults to 0.

    Returns:
        DataLoader: A DataLoader object for loading the dataset.
    """
    # get a data loader to load full dataset, not split
    features = channel_features_all.get(sessions[session_idx])
    labels = channel_labels_all.get(sessions[session_idx])
    trials = channel_trials_all.get(sessions[session_idx])
    channels = channel_channels_all.get(sessions[session_idx])
    print(f"Session {sessions[session_idx]}: Features shape: {features.shape}, Labels shape: {labels.shape}, Trials shape: {trials.shape}, Channels shape: {channels.shape}")
    # regard label, trial, channel as y, concatenate them
    y = np.column_stack((labels, trials, channels))
    assert len(features) == len(y), f"Data size mismatch for session {sessions[session_idx]}"
    # create dataset
    full_dataset = Dataset(features, y, spectrogram_size=500,
                           time_bins=config["time_bins"], library=config["library"])
    # create data loader
    full_dataloader = DataLoader(full_dataset, batch_size=config["batch_size"], shuffle=False)

    return full_dataloader

def validate_decoder(model, dataloader, supervised_criterion, device='cpu'):
    """
    Evaluate the performance of a decoder model using a given dataset.

    Args:
        model (nn.Module): Decoder model to be evaluated.
        dataloader (DataLoader): DataLoader object containing the evaluation dataset.
        supervised_criterion (nn.Module): Loss function used for evaluation.
        device (str, optional): Device to perform the evaluation ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple: Average loss, balanced accuracy, F1 score, true labels, predicted labels, trial indices, and channel indices.
    """
    model.eval()
    total_loss = []
    total_samples, correct_predictions = 0, 0
    all_true_labels = []
    all_predicted_labels = []
    all_trial_idx = []
    all_channel_idx = []

    with torch.no_grad():
        for data, label in dataloader:
            spectrograms, _ = data
            spectrograms = spectrograms.to(device)
            label = label.to(device)
            label, trial_idx, channel_idx = label[:, 0], label[:, 1], label[:, 2]
            label = label.long()

            _, output = model(spectrograms)
            loss = supervised_criterion(output, label)

            total_loss.append(loss.item())

            _, predicted = torch.max(output, 1)
            all_true_labels.extend(label.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            all_trial_idx.extend(trial_idx.cpu().numpy())
            all_channel_idx.extend(channel_idx.cpu().numpy())

            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = balanced_accuracy_score(all_true_labels, all_predicted_labels)
    f1 = f1_score(all_true_labels, all_predicted_labels, average='macro')
    return avg_loss, accuracy, f1, all_true_labels, all_predicted_labels, all_trial_idx, all_channel_idx

def validate_encoder(encoder, dataloader, device='cpu'):
    """
    Evaluate the embedding space generated by an encoder model.

    Args:
        encoder (nn.Module): Encoder model to be evaluated.
        dataloader (DataLoader): DataLoader object containing the evaluation dataset.
        device (str, optional): Device to perform the evaluation ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple: Numpy arrays containing the embeddings, labels, trial indices, and channel indices.
    """
    encoder.eval()
    embedding = []
    labels = []
    all_trial_idx = []
    all_channel_idx = []


    with torch.no_grad():
        for data, label in dataloader:
            spectrograms, _ = data
            spectrograms = spectrograms.to(device)
            z = encoder(spectrograms)
            embedding.extend(z.cpu().numpy())
            label = label.cpu().numpy()
            label, trial_idx, channel_idx = label[:, 0], label[:, 1], label[:, 2]
            labels.extend(label)
            all_trial_idx.extend(trial_idx)
            all_channel_idx.extend(channel_idx)
    return np.array(embedding), np.array(labels), np.array(all_trial_idx), np.array(all_channel_idx)





def visualize_embedding(features, labels, sessions, session_idx, output_path, trial_idx, channel_idx, acronyms_arr, is_model=False, interactive=True):
    """
    Visualizes the embedding of features using PCA for dimensionality reduction.

    This function reduces the dimensionality of the given features using PCA and creates a 2D visualization.
    It supports both interactive (using Plotly) and static (using Matplotlib) visualizations, depending on the 
    `interactive` argument.

    Parameters:
    - features: numpy array
        The feature matrix to be visualized, where rows represent samples and columns represent features.
    - labels: numpy array
        The labels corresponding to the features, used for coloring the data points.
    - sessions: list
        A list of session names or identifiers.
    - session_idx: int
        The index of the session to visualize.
    - output_path: str
        The directory path where the output visualization will be saved.
    - trial_idx: numpy array
        The trial indices corresponding to each data point, used for hover information.
    - channel_idx: numpy array
        The channel indices corresponding to each data point, used for hover information.
    - is_model: bool, optional (default=False)
        If True, the labels are adjusted to handle model-specific format.
    - interactive: bool, optional (default=True)
        If True, generates an interactive HTML plot using Plotly. If False, generates a static PNG plot using Matplotlib.

    Returns:
    - None
    
    Saves the visualization to the specified output path as either an HTML or PNG file.
    """
    # PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'PCA 1': features_pca[:, 0],
        'PCA 2': features_pca[:, 1],
        'Label': labels,
        'Trial Index': trial_idx,
        'Channel Index': channel_idx
    })

    # Define color map
    color_map = {"CA1": "orange", "CA2": "green", "CA3": "blue", "DG": "purple", "Visual Cortex": "red"}
    df['Color'] = df['Label'].map(lambda x: color_map.get(acronyms_arr[x], 'black'))
    df['Label'] = [acronyms_arr[x] for x in labels]

    if interactive:
        # Interactive plot with Plotly
        fig = px.scatter(
            df, x='PCA 1', y='PCA 2', color='Label',
            hover_data={
                'PCA 1': False,
                'PCA 2': False,
                'Trial Index': True,
                'Channel Index': True
            },
            title=f"PCA Visualization for Session {sessions[session_idx]}",
            color_discrete_map=color_map
        )
        # Save the figure as an HTML file
        if is_model:
            output_file = f"{output_path}/{sessions[session_idx]}_model_pca.html"
        else:
            output_file = f"{output_path}/{sessions[session_idx]}_pca.html"
        fig.write_html(output_file)
        print(f"Interactive plot saved to {output_file}")
    else:
        # Static plot with Matplotlib
        plt.figure(figsize=(6, 6))
        for label in np.unique(labels):
            indices = np.where(labels == label)
            plt.scatter(df.loc[indices[0], 'PCA 1'], df.loc[indices[0], 'PCA 2'], 
                        label=label, c=color_map.get(acronyms_arr[label], 'black'), s=10)
        plt.legend()
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title(f"PCA Visualization for Session {sessions[session_idx]}")

        # Save the figure as a PNG file
        if is_model:
            output_file = f"{output_path}/{sessions[session_idx]}_model_pca.png"
        else:
            output_file = f"{output_path}/{sessions[session_idx]}_pca.png"
        plt.savefig(output_file)
        print(f"Static plot saved to {output_file}")
    return

def visualize_decoder_results(all_true_labels, all_predicted_labels, all_trial_idx, all_channel_idx, session, output_path, acronyms_arr):
    """
    Visualizes the ground truth and predicted labels for decoder results.

    This function creates scatter plots for both the ground truth and predicted labels across trials and channels.
    It splits the trials into training and testing sets, and visualizes both sets as subplots, separately for 
    ground truth and predicted results. The visualization helps in comparing the true and predicted labels 
    for each session.

    Parameters:
    - all_true_labels: list
        The ground truth labels for all data points.
    - all_predicted_labels: list
        The predicted labels for all data points.
    - all_trial_idx: list
        The trial indices corresponding to each data point.
    - all_channel_idx: list
        The channel indices corresponding to each data point.
    - session: str
        The session name or identifier.
    - output_path: str
        The directory path where the output visualizations will be saved.

    Returns:
    - None
    
    Saves two sets of figures to the specified output path: one for ground truth labels and one for predicted labels, 
    each containing training and testing subplots.
    """

    trial_length = 60
    # Split trials into train and test indices
    train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)

    # Turn labels to acronyms
    all_true_labels = [acronyms_arr[label] for label in all_true_labels]
    all_predicted_labels = [acronyms_arr[label] for label in all_predicted_labels]

    # Color map
    color_map = {"CA1": "orange", "CA2": "green", "CA3": "blue", "DG": "purple", "Visual Cortex": "red"}
    all_true_colors = [color_map[label] for label in all_true_labels]
    all_predicted_colors = [color_map[label] for label in all_predicted_labels]

    ## Plot ground truth results (train and test as subplots)
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    all_trial_idx, all_channel_idx, all_true_colors = np.array(all_trial_idx), np.array(all_channel_idx), np.array(all_true_colors)

    train_mask = np.isin(all_trial_idx, train_tr_idx)
    test_mask = np.isin(all_trial_idx, test_tr_idx)

    train_trial_idx, train_channel_idx, train_true_colors = all_trial_idx[train_mask], all_channel_idx[train_mask], all_true_colors[train_mask]
    test_trial_idx, test_channel_idx, test_true_colors = all_trial_idx[test_mask], all_channel_idx[test_mask], all_true_colors[test_mask]
    
    axes[0].scatter(train_trial_idx, train_channel_idx, c=train_true_colors, s=10, marker='^')
    axes[1].scatter(test_trial_idx, test_channel_idx, c=test_true_colors, s=10, marker='o')

    # Legends for ground truth plots
    for ax in axes:
        for label in color_map:
            ax.scatter([], [], c=color_map[label], label=label)
    axes[0].scatter([], [], c='black', marker='^', label='Train Set')
    axes[1].scatter([], [], c='black', marker='o', label='Test Set')
    axes[0].legend()
    axes[1].legend()

    # Label the axes
    axes[0].set_xlabel("Trials")
    axes[0].set_ylabel("Channel Index (Depth of Probe)")
    axes[0].set_title(f"Ground Truth Training Results for Session {session}")
    axes[1].set_xlabel("Trials")
    axes[1].set_ylabel("Channel Index (Depth of Probe)")
    axes[1].set_title(f"Ground Truth Testing Results for Session {session}")

    # Save the figure
    plt.savefig(f"{output_path}/{session}_groundtruth_results.png")
    plt.close()
    print("save ground truth result")

    ## Plot predicted results (train and test as subplots)
    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    all_predicted_colors = np.array(all_predicted_colors)
    train_pred_colors, test_pred_colors = all_predicted_colors[train_mask], all_predicted_colors[test_mask]

    axes[0].scatter(train_trial_idx, train_channel_idx, c=train_pred_colors, s=10, marker='^')
    axes[1].scatter(test_trial_idx, test_channel_idx, c=test_pred_colors, s=10, marker='o')

    # Legends for predicted plots
    for ax in axes:
        for label in color_map:
            ax.scatter([], [], c=color_map[label], label=label)
    axes[0].scatter([], [], c='black', marker='^', label='Train Set')
    axes[1].scatter([], [], c='black', marker='o', label='Test Set')
    axes[0].legend()
    axes[1].legend()

    # Label the axes
    axes[0].set_xlabel("Trials")
    axes[0].set_ylabel("Channel Index (Depth of Probe)")
    axes[0].set_title(f"Predicted Training Results for Session {session}")
    axes[1].set_xlabel("Trials")
    axes[1].set_ylabel("Channel Index (Depth of Probe)")
    axes[1].set_title(f"Predicted Testing Results for Session {session}")

    # Save the figure
    plt.savefig(f"{output_path}/{session}_predicted_results.png")
    plt.close()
    print("save predicted")
def vis(data, data_type, model_name, encoder, model, i, sessions, data_path, output_path):
    """
    Visualize the results of the model
    data: str, data type
    data_type: str, data type
    model_name: str, model name
    encoder: nn.Module, encoder model
    model: nn.Module, full model
    i: int, index of session
    sessions: list, list of session names
    data_path: str, path to data
    output_path: str, path to save output
    """
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Visual Cortex'}
    acronyms_arr = np.array(['Visual Cortex', 'CA1', 'CA2', 'CA3', 'DG'])

    print(f"Visualizing results... {data} {data_type} {model_name} {sessions[i]}")
    ## config and settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sweep_config = {'spectrogram_size': 500, 'time_bins': 16, 'library': 'pytorch', 'batch_size': 64}
    ## load data
    print("Loading preprocessed data...")
    channel_features_all, channel_labels_all, channel_trials_all, channel_channels_all = load_preprocessed_data(data_path, [sessions[i]])
    features = channel_features_all.get(sessions[i])
    labels = channel_labels_all.get(sessions[i])
    trials = channel_trials_all.get(sessions[i])
    channels = channel_channels_all.get(sessions[i])
    full_dataloader = build_single_session_full_loader(channel_features_all, channel_labels_all, channel_trials_all, channel_channels_all, sweep_config, RawDataset, sessions, i)

    ### visualize feature embeddings
    print("Visualizing feature...")
    visualize_embedding(features, labels, sessions, i, output_path, trial_idx=trials, channel_idx=channels, acronyms_arr=acronyms_arr, is_model=False, interactive=True)

    ## validate encoder
    print("Validating encoder model...")
    embedding_encoder, labels_encoder, trial_idx_encoder, channel_idx_encoder = validate_encoder(encoder, full_dataloader, device=device)
    visualize_embedding(embedding_encoder, labels, sessions, i, output_path, trial_idx_encoder, channel_idx_encoder, acronyms_arr=acronyms_arr, is_model=True, interactive=True)

    ### inference 
    print("Validating decoder model...")
    supervised_criterion = nn.CrossEntropyLoss()
    avg_loss, accuracy, f1, all_true_labels, all_predicted_labels, all_trial_idx, all_channel_idx = validate_decoder(model, full_dataloader, supervised_criterion,device=device)
    print(f"Session {sessions[i]}: Average Loss: {avg_loss}, Accuracy: {accuracy}, F1: {f1}")
    ### visualize results
    visualize_decoder_results(all_true_labels, all_predicted_labels, all_trial_idx, all_channel_idx, sessions[i], output_path, acronyms_arr)

    print("visualization done")