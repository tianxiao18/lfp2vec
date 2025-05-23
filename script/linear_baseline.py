import os
import pickle
import random
import warnings
import numpy as np
import time
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, GridSearchCV, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

import torch
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blind_localization.models.SimpleLinearModel import *

warnings.filterwarnings('ignore')
from utils import *

print("Job start")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(os.getcwd())

hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Visual Cortex'}
acronyms_arr = np.array(sorted(list(hc_acronyms)))
acronyms_arr_num = np.arange(len(acronyms_arr))
acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
print(acr_dict)


#######################################################################################################################
# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='MLP baseline')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen, ibl or Neuronexus', default='Neuronexus')
    parser.add_argument('--data_type', type=str, help='Type of data to use: raw or spectrogram', default='spectrogram')
    parser.add_argument('--trial_length',  type=int, default=60, help='trial_length')
    return parser.parse_args()


args = arg_parser()
data, data_type, trial_length = args.data, args.data_type, args.trial_length
print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}")

train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)
train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=0.2, random_state=42)

if data == "Allen":
    if data_type == "raw":
        input_dim = 13
        output_dim = 5
        batch_size = 128
        lr = 0.008
    elif data_type == "spectrogram":
        input_dim = 8000
        output_dim = 5
        batch_size = 64
        lr = 0.0003
if data == "ibl":
    if data_type == "raw":
        input_dim = 13
        output_dim = 5
        batch_size = 64
        lr = 0.00537
    elif data_type == "spectrogram" or data_type == "spectrogram_preprocessed":
        input_dim = 8000
        output_dim = 5
        batch_size = 32
        lr = 4.45e-05
if data == "Neuronexus":
    if data_type == "raw":
        input_dim = 13
        output_dim = 5
        batch_size = 128
        lr = 0.008
    elif data_type == "spectrogram":
        input_dim = 8000
        output_dim = 5
        batch_size = 64
        lr = 0.0003

###########
output_path = f"../results/{data}/{data_type}/linear"
if not os.path.exists(output_path):
    os.makedirs(output_path)

sessions_list = []

if data == "Allen":
    pickle_path = "spectrogram/Allen"
    sessions_list = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
elif data == "ibl":
    sessions_list = ['15763234-d21e-491f-a01b-1238eb96d389', '1a507308-c63a-4e02-8f32-3239a07dc578',
                     '4a45c8ba-db6f-4f11-9403-56e06a33dfa4', '56956777-dca5-468c-87cb-78150432cc57',
                     '5b49aca6-a6f4-4075-931a-617ad64c219c', '73918ae1-e4fd-4c18-b132-00cb555b1ad2',
                     'b39752db-abdb-47ab-ae78-e8608bbf50ed']
    sessions_list = sessions_list
    pickle_path = f'spectrogram/ibl'
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
elif data == "Neuronexus":
    sessions_list = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    pickle_path = f'spectrogram/Neuronexus'
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Cortex'}
print(f"Sessions List: {sessions_list}")


#######################################################################################################################
# Within session
#######################################################################################################################


# def prepare_within_data(X_train, y_train, n_splits=5):
#     kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     X_fit, y_fit, groups = [], [], []
#     group_cnt = 0
#     for train_idx, val_idx in kf.split(X_train, y_train):
#         X_fit.append(np.array(X_train)[val_idx])
#         y_fit.append(np.array(y_train)[val_idx])
#         groups.extend([group_cnt] * len(val_idx))
#         group_cnt += 1
#
#     return np.vstack(X_fit), np.hstack(y_fit), np.array(groups)
#
#
# def get_within_best_params(sessions_list):
#     idx = 0
#     print(f"Fetching best parameters for within session")
#     data = pickle.load(open(f"{pickle_path}/{sessions_list[idx]}_data.pickle", 'rb'))
#     if data_type == "spectrogram":
#         X, y, trial_idx = zip(*[(np.transpose(d[0]).flatten(), d[2], d[3]) for d in data])
#     elif data_type == "raw":
#         X, y, trial_idx = zip(*[(d[0], d[1], d[2]) for d in data])
#
#     # Split data
#     train_idx = [idx for idx, val in enumerate(trial_idx) if val in train_tr_idx]
#     test_idx = [idx for idx, val in enumerate(trial_idx) if val not in train_tr_idx]
#
#     X_train, y_train = np.array(X)[train_idx], np.array(y)[train_idx]
#     X_fit, y_fit, groups = prepare_within_data(X_train, y_train)
#
#     # Train the SVM model
#     model = LogisticRegression(solver='lbfgs', max_iter=100000,
#                                class_weight='balanced', penalty='l2', n_jobs=-1, verbose=LR_verbose)
#
#     param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
#     logo = LeaveOneGroupOut()
#
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=logo, scoring='accuracy', n_jobs=-1,
#                                verbose=grid_verbosity)
#     grid_search.fit(X_fit, y_fit, groups=groups)
#
#     best_params = grid_search.best_params_
#     best_score = grid_search.best_score_
#     print(f"Best Parameters: {best_params}\nBest CV Score: {best_score}")
#     return best_params


def create_spectrogram(X, y, trial_idx):
    new_x = []
    for i in range(len(X)):
        # Provide axis = 0 for freq normalization and axis = 1 for time normalization
        new_x.append(compute_spectrogram_librosa(X[i], 0, axis=1).flatten())

    return new_x, y, trial_idx


def create_raw(X, y, trial_idx):
    sampling_rate = 1250
    data = []
    for i in range(len(X)):
        base_rms_mini = calculate_rms(X[i])
        pow_whole_mini, rms_whole_mini = calculate_power(X[i], sampling_rate)
        pow_delta_mini, rms_delta_mini = calculate_power(X[i], sampling_rate, (0, 4))
        pow_theta_mini, rms_theta_mini = calculate_power(X[i], sampling_rate, (4, 8))
        pow_alpha_mini, rms_alpha_mini = calculate_power(X[i], sampling_rate, (8, 12))
        pow_beta_mini, rms_beta_mini = calculate_power(X[i], sampling_rate, (12, 30))
        pow_gamma_mini, rms_gamma_mini = calculate_power(X[i], sampling_rate, (30, -1))

        data.append(np.array([base_rms_mini, pow_whole_mini, pow_delta_mini, pow_theta_mini, pow_alpha_mini,
                              pow_beta_mini, pow_gamma_mini, rms_whole_mini, rms_delta_mini, rms_theta_mini,
                              rms_alpha_mini, rms_beta_mini, rms_gamma_mini]).flatten())

    return data, y, trial_idx


def within_session_pipeline(i, sessions, dataset):
    print(f"Running within session pipeline for {sessions[i]}")
    # Load data
    if dataset == "Neuronexus":
        data = pickle.load(open(f"{pickle_path}/{sessions[i]}_lfp.pickle", 'rb'))
    else:
        data = pickle.load(open(f"{pickle_path}/{sessions[i]}_raw.pickle", 'rb'))
    X, y, trial_idx = zip(*[(d[0], d[1], d[2]) for d in data])
    if args.data_type == "spectrogram" or args.data_type == "spectrogram_preprocessed":
        X, y, trial_idx = create_spectrogram(X, y, trial_idx)
    elif args.data_type == "raw":
        X, y, trial_idx = create_raw(X, y, trial_idx)

    model = SimpleLinearModel(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Split data
    train_idx = [idx for idx, val in enumerate(trial_idx) if val in train_tr_idx]
    test_idx = [idx for idx, val in enumerate(trial_idx) if val not in train_tr_idx]

    X_train, y_train = np.array(X)[train_idx], np.array(y)[train_idx]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    for train_index, val_index in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train the MLP model
        for epoch in range(200):
            time_start = time.time()
            train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
            time_end = time.time()
            print(f"(Fold {fold + 1}, Epoch {epoch + 1}): Train Loss = {train_loss:.4f}, "
                  f"Train Accuracy = {train_acc:.2f}%, Time = {time_end - time_start:.2f}s")

        val_loss, val_acc, _ = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_acc:.2f}%")
        fold += 1

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(np.array(X)[test_idx], dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(np.array(y)[test_idx], dtype=torch.long).to(device)

    # Evaluate on training data
    y_pred_train = model.predict(X_train_tensor, device)
    train_pred_labels = y_pred_train.cpu().numpy()
    train_real_labels = y_train_tensor.cpu().numpy()

    # Evaluate on test data
    y_pred_test = model.predict(X_test_tensor, device)
    test_pred_labels = y_pred_test.cpu().numpy()
    test_real_labels = y_test_tensor.cpu().numpy()
    calculate_balanced_accuracy([train_real_labels], [train_pred_labels],
                                [test_real_labels], [test_pred_labels],
                                [sessions[i]], "Within Session")

    return i, (train_real_labels, train_pred_labels), (test_real_labels, test_pred_labels)


def within_session_pipeline_parallel(sessions_list):
    n_sessions = len(sessions_list)
    train_real_labels_all = [None] * n_sessions
    train_pred_labels_all = [None] * n_sessions
    test_real_labels_all = [None] * n_sessions
    test_pred_labels_all = [None] * n_sessions

    # best_params = get_within_best_params(sessions_list)

    # Run pairwise classifications in parallel
    results = Parallel(n_jobs=1, backend="multiprocessing")(
        delayed(within_session_pipeline)(i, sessions_list, data) for i in range(n_sessions)
    )

    for i, (train_real, train_pred), (test_real, test_pred) in results:
        train_real_labels_all[i] = train_real
        train_pred_labels_all[i] = train_pred
        test_real_labels_all[i] = test_real
        test_pred_labels_all[i] = test_pred

    return train_real_labels_all, train_pred_labels_all, test_real_labels_all, test_pred_labels_all


#######################################################################################################################
# Across session (Inductive) (Requires more than 2 sessions to work)
#######################################################################################################################


# def prepare_inductive_across_data(idx, sessions):
#     X_train, y_train = [], []
#     X_val, y_val = [], []
#     groups = []
#     group_cnt = 0
#     X_val_temp, y_val_temp = [], []

#     for i in range(len(sessions)):
#         if i != idx:
#             data = pickle.load(open(f"{pickle_path}/{sessions_list[idx]}_data.pickle", 'rb'))
#             if data_type == "spectrogram":
#                 X, y, trial_idx = zip(*[(np.transpose(d[0]).flatten(), d[2], d[3]) for d in data])
#             elif data_type == "raw":
#                 X, y, trial_idx = zip(*[(d[0], d[1], d[2]) for d in data])

#             train_idx = [idx for idx, val in enumerate(trial_idx) if val in train_tr_idx]

#             X_train.extend(np.array(X)[train_idx])
#             y_train.extend(np.array(y)[train_idx])
#             X_val_temp.append(np.array(X)[train_idx])
#             y_val_temp.append(np.array(y)[train_idx])

#     for i in range(len(X_val_temp)):
#         X_val.extend(X_val_temp[i])
#         y_val.extend(y_val_temp[i])
#         groups.extend([group_cnt] * len(X_val_temp[i]))
#         group_cnt += 1

#     return X_train, y_train, X_val, y_val, groups


# def get_inductive_best_params(sessions_list):
#     idx = 0
#     print(f"Fetching best parameters for inductive across session")
#     X_train, y_train, X_fit, y_fit, groups = prepare_inductive_across_data(idx, sessions_list)

#     # Train the SVM model
#     model = LogisticRegression(solver='lbfgs', max_iter=100000,
#                                class_weight='balanced', penalty='l2', n_jobs=2)

#     param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
#     logo = LeaveOneGroupOut()

#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=logo, scoring='accuracy', n_jobs=-1,
#                                verbose=grid_verbosity)
#     grid_search.fit(X_fit, y_fit, groups=groups)

#     best_params = grid_search.best_params_
#     best_score = grid_search.best_score_
#     print(f"Best Parameters: {best_params}\nBest CV Score: {best_score}")
#     return best_params


# def inductive_across_session_pipeline(idx, sessions, best_params):
#     print(f"Running inductive across session pipeline for {sessions[idx]}")

#     # Load data
#     def load_session_data(session_idx):
#         data = pickle.load(open(f"{pickle_path}/{sessions[session_idx]}_data.pickle", 'rb'))
#         if data_type == "spectrogram":
#             X, y, trial_idx = zip(*[(np.transpose(d[0]).flatten(), d[2], d[3]) for d in data])
#         elif data_type == "raw":
#             X, y, trial_idx = zip(*[(d[0], d[1], d[2]) for d in data])
#         return np.array(X), np.array(y), trial_idx

#     # Load data
#     X, y, trial_idx = load_session_data(idx)
#     test_idx = [idx for idx, val in enumerate(trial_idx) if val not in train_tr_idx]
#     X_test, y_test = X[test_idx], y[test_idx]

#     X_train_main, y_train_main = [], []

#     fold_count = 0
#     for i in range(len(sessions)):
#         if i != idx:
#             X_train, y_train, trial_idx = load_session_data(i)
#             train_idx = [idx for idx, val in enumerate(trial_idx) if val in train_tr_idx]
#             X_train, y_train = X_train[train_idx], y_train[train_idx]
#             X_train_main.extend(X_train)
#             y_train_main.extend(y_train)

#     # Train the SVM model
#     model = LogisticRegression(solver='lbfgs', max_iter=100000,
#                                class_weight='balanced', penalty='l2', n_jobs=-1)

#     model.set_params(**best_params)
#     model.fit(np.array(X_train_main), np.array(y_train_main))

#     # Evaluate on training data
#     y_pred_train = model.predict(np.array(X_train_main))
#     train_pred_labels = np.array(y_pred_train)
#     train_real_labels = np.array(y_train_main)

#     # Evaluate on test data
#     y_pred_test = model.predict(np.array(X_test))
#     test_pred_labels = np.array(y_pred_test)
#     test_real_labels = np.array(y_test)

#     return idx, (train_real_labels, train_pred_labels), (test_real_labels, test_pred_labels)

def inductive_across_session_pipeline(idx, sessions, dataset):
    print(f"Running inductive across session pipeline for {sessions[idx]}")

    # Load data
    def load_session_data(i, sessions):
        if dataset == "Neuronexus":
            data = pickle.load(open(f"{pickle_path}/{sessions[i]}_lfp.pickle", 'rb'))
        else:
            data = pickle.load(open(f"{pickle_path}/{sessions[i]}_raw.pickle", 'rb'))
        X, y, trial_idx = zip(*[(d[0], d[1], d[2]) for d in data])
        if args.data_type == "spectrogram" or args.data_type == "spectrogram_preprocessed":
            X, y, trial_idx = create_spectrogram(X, y, trial_idx)
        elif args.data_type == "raw":
            X, y, trial_idx = create_raw(X, y, trial_idx)
        return np.array(X), np.array(y), trial_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, trial_idx = load_session_data(idx, sessions)
    test_idx = [idx for idx, val in enumerate(trial_idx) if val not in train_tr_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    X_train_main, y_train_main = [], []
    folds_data = {i: {'X_train': [], 'y_train': [], 'X_val': [], 'y_val': []} for i in range(len(sessions) - 1)}

    model = SimpleLinearModel(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)

    fold_count = 0
    sessions_except_test = [sess for sess in sessions if sess != sessions[idx]]
    sessions_except_test = random.sample(sessions_except_test, 4)
    for i in range(len(sessions_except_test)):
        if i != idx:
            X_train, y_train, trial_idx = load_session_data(i, sessions_except_test)
            train_idx = [idx for idx, val in enumerate(trial_idx) if val in train_tr_idx]
            X_train, y_train = X_train[train_idx], y_train[train_idx]
            X_train_main.extend(X_train)
            y_train_main.extend(y_train)

            folds_data[fold_count]['X_val'].extend(X_train)
            folds_data[fold_count]['y_val'].extend(y_train)
            fold_count += 1

    for i in range(len(sessions_except_test) - 1):
        for j in range(len(sessions_except_test) - 1):
            if i != j:
                folds_data[i]['X_train'].extend(folds_data[j]['X_val'])
                folds_data[i]['y_train'].extend(folds_data[j]['y_val'])

    for fold in range(len(sessions_except_test) - 1):
        X_train_fold = folds_data[fold]['X_train']
        y_train_fold = folds_data[fold]['y_train']
        X_val_fold = folds_data[fold]['X_val']
        y_val_fold = folds_data[fold]['y_val']

        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train the MLP model
        for epoch in range(100):
            time_start = time.time()
            train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
            time_end = time.time()
            print(f"(Fold {fold + 1}, Epoch {epoch + 1}): Train Loss = {train_loss:.4f}, "
                  f"Train Accuracy = {train_acc:.2f}%, Time = {time_end - time_start:.2f}s")

        val_loss, val_acc, _ = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_acc:.2f}%")

    X_train_tensor = torch.tensor(X_train_main, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_main, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Evaluate on training data
    y_pred_train = model.predict(X_train_tensor, device)
    train_pred_labels = y_pred_train.cpu().numpy()
    train_real_labels = y_train_tensor.cpu().numpy()

    # Evaluate on test data
    y_pred_test = model.predict(X_test_tensor, device)
    test_pred_labels = y_pred_test.cpu().numpy()
    test_real_labels = y_test_tensor.cpu().numpy()

    return idx, (train_real_labels, train_pred_labels), (test_real_labels, test_pred_labels)


def inductive_across_session_pipeline_parallel(sessions_list):
    n_sessions = len(sessions_list)
    train_real_labels_all = [None] * n_sessions
    train_pred_labels_all = [None] * n_sessions
    test_real_labels_all = [None] * n_sessions
    test_pred_labels_all = [None] * n_sessions

    # best_params = get_inductive_best_params(sessions_list)

    # Run pairwise classifications in parallel
    results = Parallel(n_jobs=1, backend="multiprocessing")(
        delayed(inductive_across_session_pipeline)(i, sessions_list, data) for i in range(n_sessions)
    )

    for i, (train_real, train_pred), (test_real, test_pred) in results:
        train_real_labels_all[i] = train_real
        train_pred_labels_all[i] = train_pred
        test_real_labels_all[i] = test_real
        test_pred_labels_all[i] = test_pred

    return train_real_labels_all, train_pred_labels_all, test_real_labels_all, test_pred_labels_all


if __name__ == "__main__":
    print("Running within session pipeline")

    train_real_labels_all, train_pred_labels_all, test_real_labels_all, test_pred_labels_all = within_session_pipeline_parallel(
        sessions_list)

    calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
                                test_pred_labels_all, sessions_list, "Within Session")

    with open(f"{output_path}/within_train_real_labels.pickle", 'wb') as f:
        pickle.dump(train_real_labels_all, f)

    with open(f"{output_path}/within_train_pred_labels.pickle", 'wb') as f:
        pickle.dump(train_pred_labels_all, f)

    with open(f"{output_path}/within_test_real_labels.pickle", 'wb') as f:
        pickle.dump(test_real_labels_all, f)

    with open(f"{output_path}/within_test_pred_labels.pickle", 'wb') as f:
        pickle.dump(test_pred_labels_all, f)

    #######################################################################################################################

    print("Running inductive across session pipeline")

    train_real_labels_all, train_pred_labels_all, test_real_labels_all, test_pred_labels_all = inductive_across_session_pipeline_parallel(
        sessions_list)

    calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
                                test_pred_labels_all, sessions_list, "Inductive Across Session")

    with open(f"{output_path}/inductive_train_real_labels.pickle", 'wb') as f:
        pickle.dump(train_real_labels_all, f)

    with open(f"{output_path}/inductive_train_pred_labels.pickle", 'wb') as f:
        pickle.dump(train_pred_labels_all, f)

    with open(f"{output_path}/inductive_test_real_labels.pickle", 'wb') as f:
        pickle.dump(test_real_labels_all, f)

    with open(f"{output_path}/inductive_test_pred_labels.pickle", 'wb') as f:
        pickle.dump(test_pred_labels_all, f)

    #######################################################################################################################