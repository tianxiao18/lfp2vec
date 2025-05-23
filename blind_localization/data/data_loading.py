import os
import h5py
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import json
import pandas as pd

def load_channel_map(file_path):
    filepath = os.path.abspath(file_path)

    with h5py.File(filepath, 'r') as file:
        x_coord = file['xcoords'][:]
        y_coord = file['ycoords'][:]

    channel_map = np.concatenate([x_coord, y_coord], axis=1)
    channel_map = np.concatenate([channel_map[i*128:i*128+128][np.lexsort((channel_map[i*128:i*128+128, 0], channel_map[i*128:i*128+128, 1]))] for i in range(8)], axis=0)

    xi = np.insert(channel_map, 1, np.ones(len(channel_map)), axis=1)
    D = cdist(xi, xi, 'euclidean')

    return channel_map, D


def custom_train_test_split(indices, ridx_map, train_size, test_size=0.2, random_state=66, min_samples_per_class=2):
    """
    Try to satisfy the following conditions in order
    1. Hold out 20% fixed data for each class for testing
    2. Ensure at least 2 training samples is available each class
    3. Attempt to meet the specified train size for each class with any remaining samples
    """
    regions = np.unique(ridx_map[:, 1])
    np.random.seed(random_state)
    indices_train = []
    indices_test = []

    for r in regions:
        idx_region_r = indices[ridx_map[:, 1] == r]
        n_samples = len(idx_region_r)

        if n_samples < min_samples_per_class:
            raise ValueError(f"Not enough samples for class {r}")
        
        np.random.shuffle(idx_region_r)
        cls_test_indices = idx_region_r[:int(test_size*n_samples)]
        
        idx_left = np.setdiff1d(idx_region_r, cls_test_indices)
        adjusted_train_size = min(max(min_samples_per_class, int(train_size * n_samples)), n_samples)
        cls_train_indices = np.random.choice(idx_left, adjusted_train_size, replace=False)

        indices_train.extend(cls_train_indices)
        indices_test.extend(cls_test_indices)

    return indices_train, indices_test


def custom_cv_split(indices, ridx_map, train_size, test_size=0.2, random_state=66, min_samples_per_class=2, n_splits=5):
    regions = np.unique(ridx_map[:, 1])
    np.random.seed(random_state)

    for fold in range(n_splits):
        indices_train = []
        indices_test = []

        for r in regions:
            idx_region_r = indices[ridx_map[:, 1] == r]
            n_samples = len(idx_region_r)

            if n_samples < min_samples_per_class:
                raise ValueError(f"Not enough samples for class {r}")
            
            np.random.shuffle(idx_region_r)
            fold_size = n_samples // n_splits
            start, end = fold * fold_size, (fold + 1) * fold_size if fold != n_splits - 1 else n_samples
            
            cls_test_indices = idx_region_r[start:end]
            cls_train_indices = np.setdiff1d(idx_region_r, cls_test_indices)

            indices_train.extend(cls_train_indices)
            indices_test.extend(cls_test_indices)

        yield indices_train, indices_test


def load_session_data(json_file_path, session_name):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        return data.get(session_name)
    

def get_train_test_indices(channel_ridx_map, random_state=66, train_size=0.8, test_size=0.2, custom_split=False, min_samples_per_class=2):
    # find (channel, region) for region = unk, idx_test is index in channel_ridx_map, not actual channel idx
    idx_test = np.arange(len(channel_ridx_map))[channel_ridx_map[:, 1] == 5]

    mask = np.isin(np.arange(len(channel_ridx_map)), idx_test)
    masked_channel_region_map = channel_ridx_map[~mask]

    indices = np.arange(len(masked_channel_region_map))

    if custom_split:
        idx_train, idx_val = custom_train_test_split(indices, masked_channel_region_map, train_size, test_size=test_size, random_state=random_state, min_samples_per_class=min_samples_per_class)
    else:
        idx_train, idx_val = train_test_split(indices, test_size=1-train_size, random_state=random_state)

    channel_idx_train = channel_ridx_map[~mask][:, 0][idx_train]
    channel_idx_val = channel_ridx_map[~mask][:, 0][idx_val]
    channel_idx_test = channel_ridx_map[mask][:, 0]
    indices = idx_train, idx_val, idx_test

    return channel_idx_train, channel_idx_val, channel_idx_test, indices


def load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, 
                         train_size=0.8, val_size=0.2, custom_split=False, transferred_channel_idx_train=None, min_samples_per_class=2):
    """
    Here we load train test data given input data f, D, with output label r
    by train test spliting channel_ridx_map and use it to index f, D and r
    """
    channel_idx_train, channel_idx_val, channel_idx_test, indices = get_train_test_indices(channel_ridx_map, random_state=random_state, train_size=train_size, test_size=val_size, custom_split=custom_split, min_samples_per_class=min_samples_per_class)
    idx_train, idx_val, idx_test = indices
    mask = np.isin(np.arange(len(channel_ridx_map)), idx_test)

    channel_indices = {
        'train': channel_ridx_map[~mask][:, 0][idx_train],
        'val': channel_ridx_map[~mask][:, 0][idx_val],
        'test': channel_ridx_map[mask][:, 0]
    }

    channel_features_sets = {key: channel_features[indices] for key, indices in channel_indices.items()}

    distances_sets = {
        'train': D[np.ix_(channel_idx_train, channel_idx_train)],
        'val': D[np.ix_(channel_idx_val, channel_idx_train)],
        'test': D[np.ix_(channel_idx_test, channel_idx_train)]
    }

    if transferred_channel_idx_train is not None:
        distances_sets = {
        'train': D[np.ix_(channel_idx_train, transferred_channel_idx_train)],
        'val': D[np.ix_(channel_idx_val, transferred_channel_idx_train)],
        'test': D[np.ix_(channel_idx_test, transferred_channel_idx_train)]
        }

    X_train = (channel_features_sets['train'], distances_sets['train'])
    X_val = (channel_features_sets['val'], distances_sets['val'])
    X_test = (channel_features_sets['test'], distances_sets['test'])

    y_train = channel_ridx_map[~mask][idx_train][:, 1]
    y_val = channel_ridx_map[~mask][idx_val][:, 1]

    return X_train, X_val, X_test, y_train, y_val

def load_results(results_path, field='test_acc'):
    df = pd.read_csv(results_path)

    within_df = df[df['Name'].str.contains('within')]
    across_df = df[df['Name'].str.contains('across')]

    within_accs = within_df[field].to_numpy()[::-1]
    across_accs = across_df[field].to_numpy()[::-1]

    return within_accs, across_accs