import numpy as np
from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt
import random

def compute_similarity(raw_signal):
    """
    Here we compute cross correlation between each channel to another
    Input: n_channels * n_samples
    Output: n_channels * n_channels
    """
    norms = np.linalg.norm(raw_signal, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1, norms)
    normalized_signal = raw_signal / safe_norms
    normalized_signal[norms[:, 0] == 0] = 0

    corr_table = np.dot(normalized_signal, normalized_signal.T)
    return corr_table

def compute_edge_index(raw_signal, edge_threshold=0.7):
    adj_matrix = compute_similarity(raw_signal)
    np.fill_diagonal(adj_matrix, 0)
    binary_adj_matrix = np.where(adj_matrix > edge_threshold, 1, 0)
    edge_index = np.array(np.nonzero(binary_adj_matrix))
    return edge_index

def threshold_edge_index(raw_signal, k=5):
    adj_matrix = compute_similarity(raw_signal)
    np.fill_diagonal(adj_matrix, 0)
    
    num_nodes = adj_matrix.shape[0]
    edge_row = []
    edge_col = []

    for i in range(num_nodes):
        top_k_indices = np.argpartition(-adj_matrix[i], k)[:k]
        for j in top_k_indices:
            edge_row.append(i)
            edge_col.append(j)

    edge_index = np.array([edge_row, edge_col])
    return edge_index

def build_single_session_graph_dataloader(channel_features_all, channel_labels_all, indices, session_config, sweep_config, GraphDataset, session_idx=0, edge_threshold=0.7, batch_size=4, channel_indices=None):
    idx_train, idx_val, idx_test = indices # trial_idx
    channel_idx_train, channel_idx_val, channel_idx_test = [np.tile(channel_indices, (len(idx), 1)) for idx in indices] if session_config['channel_idx'] else (None, None, None)
    n_columns, n_channels = 8, 1024
    n_rows = n_channels // n_columns
    global_position = torch.tensor([[x, y] for x in range(n_columns) for y in range(n_rows)]) if session_config['channel_idx'] else None
    
    X_train, y_train = channel_features_all[session_idx][idx_train], channel_labels_all[session_idx]
    X_val, y_val = channel_features_all[session_idx][idx_val], channel_labels_all[session_idx]
    X_test, y_test = channel_features_all[session_idx][idx_test], channel_labels_all[session_idx]

    A_train = [threshold_edge_index(x_train, 5) for x_train in X_train]
    A_val = [threshold_edge_index(x_val, 5) for x_val in X_val]
    A_test = [threshold_edge_index(x_test, 5) for x_test in X_test]

    train_dataset = GraphDataset(X_train, A_train, y_train, sampling_rate=session_config['sampling_rate'], global_positions=global_position, channel_indices=channel_idx_train)
    val_dataset = GraphDataset(X_val, A_val, y_val, sampling_rate=session_config['sampling_rate'], global_positions=global_position, channel_indices=channel_idx_val)
    test_dataset = GraphDataset(X_test, A_test, y_test, sampling_rate=session_config['sampling_rate'], global_positions=global_position, channel_indices=channel_idx_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def build_multi_session_graph_dataloader(channel_features_all, channel_labels_all, indices, sessions, session_config, sweep_config, GraphDataset, test_session_idx=0, edge_threshold=0.7, batch_size=2, inductive=True):
    idx_train, idx_val, idx_test = indices
    
    X_test = channel_features_all[test_session_idx][idx_test]
    y_test = channel_labels_all[test_session_idx]
    idx_test = np.array(idx_test)

    if inductive:
        assert(len(sessions) >= 3)
        val_session_idx = random.choice([i for i in range(len(sessions)) if i != test_session_idx])
        X_train = np.concatenate([channel_features_all[j][~idx_test] for j in range(len(sessions)) if j != test_session_idx and j != val_session_idx], axis=1)
        y_train = np.hstack([channel_labels_all[j] for j in range(len(sessions)) if j != test_session_idx and j != val_session_idx])

        X_val = channel_features_all[val_session_idx][~idx_test]
        y_val = channel_labels_all[val_session_idx]
    else:
        X_train = np.concatenate([channel_features_all[j][idx_train] for j in range(len(sessions)) if j != test_session_idx], axis=1)
        y_train = np.hstack(channel_labels_all[:test_session_idx]+channel_labels_all[test_session_idx+1:])
    
        X_val = np.concatenate([channel_features_all[j][idx_val] for j in range(len(sessions)) if j != test_session_idx], axis=1)
        y_val = np.hstack(channel_labels_all[:test_session_idx]+channel_labels_all[test_session_idx+1:])

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    A_train = [compute_edge_index(x_train, edge_threshold) for x_train in X_train]
    A_val = [compute_edge_index(x_val, edge_threshold) for x_val in X_val]
    A_test = [compute_edge_index(x_test, edge_threshold) for x_test in X_test]

    train_dataset = GraphDataset(X_train, A_train, y_train, sampling_rate=session_config['sampling_rate'])
    val_dataset = GraphDataset(X_val, A_val, y_val, sampling_rate=session_config['sampling_rate'])
    test_dataset = GraphDataset(X_test, A_test, y_test, sampling_rate=session_config['sampling_rate'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def build_multi_session_correlation_dataloader(channel_features_all, channel_labels_all, indices, sessions, session_config, sweep_config, GraphDataset, test_session_idx=0, edge_threshold=0.7, batch_size=2, inductive=True):
    idx_train, idx_val, idx_test = indices
    
    X_test = channel_features_all[test_session_idx][idx_test]
    y_test = channel_labels_all[test_session_idx]
    idx_test = np.array(idx_test)

    assert(len(sessions) >= 3)
    val_session_idx = random.choice([i for i in range(len(sessions)) if i != test_session_idx])
    X_train = np.concatenate([channel_features_all[j][~idx_test] for j in range(len(sessions)) if j != test_session_idx and j != val_session_idx], axis=1)
    y_train = np.hstack([channel_labels_all[j] for j in range(len(sessions)) if j != test_session_idx and j != val_session_idx])

    X_val = channel_features_all[val_session_idx][~idx_test]
    y_val = channel_labels_all[val_session_idx]

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    A_train = [compute_similarity(x_train) for x_train in X_train]
    A_val = [compute_similarity(x_val) for x_val in X_val]
    A_test = [compute_similarity(x_test) for x_test in X_test]

    train_dataset = GraphDataset(X_train, A_train, y_train, sampling_rate=session_config['sampling_rate'])
    val_dataset = GraphDataset(X_val, A_val, y_val, sampling_rate=session_config['sampling_rate'])
    test_dataset = GraphDataset(X_test, A_test, y_test, sampling_rate=session_config['sampling_rate'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader