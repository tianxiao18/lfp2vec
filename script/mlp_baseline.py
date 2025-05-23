import os
import pickle
import random
import warnings
import numpy as np
import time
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, GridSearchCV, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

import torch
import argparse
import sys

from blind_localization.models.MLP import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blind_localization.models.SimpleLinearModel import *

warnings.filterwarnings('ignore')
from utils import *

print("Job start")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(os.getcwd())


#######################################################################################################################
# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='MLP baseline')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen, Neuronexus or ibl', default="Monkey")
    parser.add_argument('--data_type', type=str, help='Type of data to use: raw or spectrogram', default="raw")
    parser.add_argument('--trial_length', type=int, default=60, help='trial_length')
    return parser.parse_args()


args = arg_parser()
data, data_type, trial_length = args.data, args.data_type, args.trial_length
print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}")

hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Visual Cortex'} if data != "Monkey" else \
    {'Basal_Ganglia', 'Suppl_Motor_Area', 'Primary_Motor_Cortex'}
acronyms_arr = np.array((list(hc_acronyms)))
acronyms_arr_num = np.arange(len(acronyms_arr))
acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
print(acr_dict)

train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)
train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=0.25, random_state=42)

if data == "Allen":
    if data_type == "raw":
        input_dim = 13
        hidden_dim = 64
        output_dim = 5
        batch_size = 128
        lr = 0.008
    elif data_type == "spectrogram":
        input_dim = 8000
        hidden_dim = 512
        output_dim = 5
        batch_size = 64
        lr = 0.0003
if data == "ibl":
    if data_type == "raw":
        input_dim = 13
        hidden_dim = 32
        output_dim = 5
        batch_size = 64
        lr = 0.00537
    elif data_type == "spectrogram" or data_type == "spectrogram_preprocessed":
        input_dim = 8000
        hidden_dim = 512
        output_dim = 5
        batch_size = 32
        lr = 4.45e-05
if data == "Neuronexus":
    if data_type == "raw":
        input_dim = 13
        hidden_dim = 64
        output_dim = 5
        batch_size = 128
        lr = 0.008
    elif data_type == "spectrogram":
        input_dim = 8000
        hidden_dim = 512
        output_dim = 5
        batch_size = 64
        lr = 0.0003
if data == "Monkey":
    if data_type == "raw":
        input_dim = 13
        hidden_dim = 64
        output_dim = 3
        batch_size = 128
        lr = 0.008
    elif data_type == "spectrogram":
        input_dim = 8000
        hidden_dim = 512
        output_dim = 3
        batch_size = 64
        lr = 0.0003

###########
output_path = f"../results/{data}/{data_type}/MLP"
if not os.path.exists(output_path):
    os.makedirs(output_path)

sessions_list = []

if data == "Allen":
    pickle_path = "spectrogram/Allen"
    #             '715093703', '719161530', '721123822', '743475441', '744228101', '746083955', '750332458', '750749662',
    #             '751348571', '754312389', '754829445', '755434585', '756029989', '757216464', '757970808', '758798717',
    #             '759883607', '760345702', '761418226', '762602078', '763673393', '766640955', '767871931', '768515987',
    #             '771160300', '771990200', '773418906', '774875821', '778240327', '778998620', '779839471', '781842082',
    #             '786091066', '787025148', '789848216', '791319847', '793224716', '794812542', '797828357', '798911424',
    #             '799864342', '816200189', '819186360', '819701982', '821695405', '829720705', '831882777', '835479236',
    #             '839068429', '840012044', '847657808']
    sessions_list = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
elif data == "ibl":
    sessions_list = ['0802ced5-33a3-405e-8336-b65ebc5cb07c_probe00',
                     '0802ced5-33a3-405e-8336-b65ebc5cb07c_probe01',
                     '0a018f12-ee06-4b11-97aa-bbbff5448e9f_probe00',
                     '3638d102-e8b6-4230-8742-e548cd87a949_probe01',
                     '5dcee0eb-b34d-4652-acc3-d10afc6eae68_probe00',
                     'd2832a38-27f6-452d-91d6-af72d794136c_probe00',
                     '54238fd6-d2d0-4408-b1a9-d19d24fd29ce_probe00']
    sessions_list = sessions_list
    pickle_path = "spectrogram/Allen"
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
elif data == "Neuronexus":
    sessions_list = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02']
    pickle_path = "spectrogram/Allen"
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Cortex'}
elif data == "Monkey":
    sessions_list = ['221007', '221104', '221216']
    pickle_path = "spectrogram/Allen"
    hc_acronyms = {'Basal_Ganglia', 'Suppl_Motor_Area', 'Primary_Motor_Cortex'}
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
    for i in tqdm(range(len(X))):
        # Provide axis = 0 for freq normalization and axis = 1 for time normalization
        new_x.append(compute_spectrogram_librosa(X[i], 0, axis=1).flatten())

    return new_x, y, trial_idx


def create_raw(X, y, trial_idx):
    sampling_rate = 1250 if args.data != "Monkey" else 2500
    data = []
    for i in tqdm(range(len(X))):
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
    results = Parallel(n_jobs=-1, backend="multiprocessing")(
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
    label_json = {"0": "CA1", "1": "CA2", "2": "CA3", "3": "DG", "4": "VIS"} if args.data != "Monkey" else \
        {"0": "Basal_Ganglia", "1": "Suppl_Motor_Area", "2": "Primary_Motor_Cortex"}

    # Load data
    def load_session_data(i, sessions):
        if dataset == "Monkey":
            data = pickle.load(open(f"{pickle_path}/{sessions[i]}.pickle", 'rb'))
        elif dataset == "Neuronexus":
            data = pickle.load(open(f"{pickle_path}/{sessions[i]}_lfp.pickle", 'rb'))
        else:
            data = pickle.load(open(f"{pickle_path}/{sessions[i]}_raw.pickle", 'rb'))
        X, y, trial_idx, chan_id = zip(*[(d[0], d[1], d[2], d[3]) for d in data])
        y = [int(label.replace("imec", "")) for label in y]

        non_zero_indices = [i for i, x in enumerate(X) if not np.all(x == 0)]
        X = np.array(X)[non_zero_indices]
        y = np.array(y)[non_zero_indices]
        trial_idx = np.array(trial_idx)[non_zero_indices]
        chan_id = np.array(chan_id)[non_zero_indices]
        X = np.array(X, dtype=np.float32)

        if args.data_type == "spectrogram" or args.data_type == "spectrogram_preprocessed":
            X, y, trial_idx = create_spectrogram(X, y, trial_idx)
        elif args.data_type == "raw":
            X, y, trial_idx = create_raw(X, y, trial_idx)

        assert len(X) == len(y) == len(trial_idx) == len(chan_id), \
            "Length mismatch between X, y, trial_idx, and chan_id"

        return np.array(X), np.array(y), trial_idx, chan_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_sessions = [sessions[idx]]
    X_test, y_test, test_trial_chans = [], [], []
    for test_sess in test_sessions:
        X, y, trial_idx, chan_id = load_session_data(sessions.index(test_sess), sessions)
        test_idx = [i for i, val in enumerate(trial_idx) if val in test_tr_idx]
        X_test.append(X[test_idx])
        y_test.append(y[test_idx])
        test_trial_chans.append(np.array([f"{test_sess}_{trial_idx[i]}_{chan_id[i]}"
                                          for i in range(len(trial_idx))])[test_idx])
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    test_trial_chans = np.concatenate(test_trial_chans)
    test_sess_labels = [f"{test_sess}_{label_json[str(label)]}" for label in y_test]

    random.seed(42)
    sessions_except_test = [ses for ses in sessions if ses not in test_sessions]
    train_sessions = random.sample(sessions_except_test, int(len(sessions_except_test) * 0.8)) if args.data != "Monkey" else \
        sessions_except_test
    val_sessions = [ses for ses in sessions_except_test if ses not in train_sessions] if args.data != "Monkey" else \
        [sessions_except_test[-1]]
    print(f"Train Sessions: {train_sessions}, Val Sessions: {val_sessions}, Test Session: {test_sessions}")

    X_train, y_train, train_trial_chans = [], [], []
    for train_sess in train_sessions:
        X, y, trial_idx, chan_id = load_session_data(sessions.index(train_sess), sessions)
        train_idx = [i for i, val in enumerate(trial_idx) if val in train_tr_idx]
        X_train.append(X[train_idx])
        y_train.append(y[train_idx])
        train_trial_chans.append(np.array([f"{train_sess}_{trial_idx[i]}_{chan_id[i]}"
                                            for i in range(len(trial_idx))])[train_idx])
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    train_trial_chans = np.concatenate(train_trial_chans)
    train_sess_labels = [f"{train_sess}_{label_json[str(label)]}" for label in y_train]

    X_val, y_val, val_trial_chans = [], [], []
    for val_sess in val_sessions:
        X, y, trial_idx, chan_id = load_session_data(sessions.index(val_sess), sessions)
        val_idx = [i for i, val in enumerate(trial_idx) if val in val_tr_idx]
        X_val.append(X[val_idx])
        y_val.append(y[val_idx])
        val_trial_chans.append(np.array([f"{val_sess}_{trial_idx[i]}_{chan_id[i]}"
                                            for i in range(len(trial_idx))])[val_idx])
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)
    val_trial_chans = np.concatenate(val_trial_chans)
    val_sess_labels = [f"{val_sess}_{label_json[str(label)]}" for label in y_val]

    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    print(model)
    model_config = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'batch_size': batch_size,
        'lr': lr
    }

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the MLP model
    patience = 20
    best_val_acc = 0
    for epoch in range(300):
        time_start = time.time()
        train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_acc, _ = evaluate_model(model, val_dataloader, criterion, device)
        time_end = time.time()
        print(f"Epoch {epoch + 1}): Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy = {val_acc:.2f}%, Time = {time_end - time_start:.2f}s")
        if val_acc < best_val_acc:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break
        else:
            patience = 20
            best_val_acc = val_acc
            print(f"Best validation accuracy: {best_val_acc:.2f}%")

    torch.save(model.state_dict(), f"{output_path}/model_{sessions[idx]}_across.pth")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

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

    # Evaluate on training data
    y_pred_train = model.predict(X_train_tensor, device)
    y_pred_train_logits = model(X_train_tensor.to(device))
    train_pred_labels = y_pred_train.cpu().numpy()
    train_real_labels = y_train_tensor.cpu().numpy()
    train_acc = (train_pred_labels == train_real_labels).mean()

    # Evaluate on validation data
    y_pred_val = model.predict(X_val_tensor, device)
    y_pred_val_logits = model(X_val_tensor.to(device))
    val_pred_labels = y_pred_val.cpu().numpy()
    val_real_labels = y_val_tensor.cpu().numpy()
    val_acc = (val_pred_labels == val_real_labels).mean()

    # Evaluate on test data
    y_pred_test = model.predict(X_test_tensor, device)
    y_pred_test_logits = model(X_test_tensor.to(device))
    test_pred_labels = y_pred_test.cpu().numpy()
    test_real_labels = y_test_tensor.cpu().numpy()
    test_acc = (test_pred_labels == test_real_labels).mean()

    train_embeddings, train_embedding_labels = get_embeddings(model, train_dataloader, device)
    val_embeddings, val_embedding_labels = get_embeddings(model, val_dataloader, device)
    test_embeddings, test_embedding_labels = get_embeddings(model, test_dataloader, device)

    file_path = os.path.join(output_path, f'{sessions[idx]}_results.pickle')
    file_obj = {
        'train_logits': y_pred_train_logits,  # Predicted logits for training set
        'train_labels': train_real_labels,  # True labels for training set
        'train_acc': train_acc,  # Accuracy for training set
        'train_trial_chans': train_trial_chans,  # Trial and channel IDs for training set ("<Session>_<Trial>_<Chan>")
        'train_sess_labels': train_sess_labels,  # Session labels for training set ("<Session>_<Label>")
        'val_logits': y_pred_val_logits,  # Predicted logits for validation set
        'val_labels': val_real_labels,  # True labels for validation set
        'val_acc': val_acc,  # Accuracy for validation set
        'val_trial_chans': val_trial_chans,  # Trial and channel IDs for validation set ("<Session>_<Trial>_<Chan>")
        'val_sess_labels': val_sess_labels,  # Session labels for validation set ("<Session>_<Label>")
        'test_logits': y_pred_test_logits,  # Predicted logits for test set
        'test_labels': test_real_labels,  # True labels for test set
        'test_acc': test_acc,  # Accuracy for test set
        'test_trial_chans': test_trial_chans,  # Trial and channel IDs for test set ("<Session>_<Trial>_<Chan>")
        'test_sess_labels': test_sess_labels,  # Session labels for test set ("<Session>_<Label>")
        'train_label_counts': train_label_counts,  # Label counts for training set
        'train_chance_accuracy': train_chance_accuracy,  # Chance accuracy for training set
        'val_label_counts': val_label_counts,  # Label counts for validation set
        'val_chance_accuracy': val_chance_accuracy,  # Chance accuracy for validation set
        'test_label_counts': test_label_counts,  # Label counts for test set
        'test_chance_accuracy': test_chance_accuracy,  # Chance accuracy for test set
        'train_embeddings': train_embeddings,  # Embeddings for training set
        'val_embeddings': val_embeddings,  # Embeddings for validation set
        'test_embeddings': test_embeddings,  # Embeddings for test set
        'train_embedding_labels': train_embedding_labels,  # Labels for training set embeddings
        'val_embedding_labels': val_embedding_labels,  # Labels for validation set embeddings
        'test_embedding_labels': test_embedding_labels,  # Labels for test set embeddings
        'mlp_config': model_config,  # Model configuration
        'best_ckpt_path': f"{output_path}/model_{sessions[idx]}_across.pth"  # Path to the best model
    }
    with open(file_path, 'wb') as f:
        pickle.dump(file_obj, f)
        print(f"Session {sessions[idx]} results saved to {file_path}")
        print(f"Train accuracy: {train_acc}, Validation accuracy: {val_acc}, Test accuracy: {test_acc}")
        print(f"Train label counts: {train_label_counts}, "
              f"Validation label counts: {val_label_counts}, "
              f"Test label counts: {test_label_counts}")
        print(f"Train chance accuracy: {train_chance_accuracy}, "
              f"Validation chance accuracy: {val_chance_accuracy}, "
              f"Test chance accuracy: {test_chance_accuracy}")
        print(f"Model config: {model_config}, "
              f"Model path: {output_path}/model_{sessions[idx]}_across.pth")

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
    ######### Hyperparameter optimization #########
    # Run Optuna for each session
    # best_params_list = []
    # for i in range(len(sessions_list)):
    #     best_params, best_val_acc = run_optuna(i, sessions_list)
    #     print(f"Session: {sessions_list[i]}, Best Parameters: {best_params}, Best Validation Accuracy: {best_val_acc}")
    #     best_params_list.append(best_params)
    # print("Best hyperparameters for all sessions:")
    # print(best_params_list)

    ######### Within session pipeline #########
    # print("Running within session pipeline")
    #
    # train_real_labels_all, train_pred_labels_all, test_real_labels_all, test_pred_labels_all = within_session_pipeline_parallel(
    #     sessions_list)
    #
    # calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
    #                             test_pred_labels_all, sessions_list, "Within Session")
    #
    # with open(f"{output_path}/within_train_real_labels.pickle", 'wb') as f:
    #     pickle.dump(train_real_labels_all, f)
    #
    # with open(f"{output_path}/within_train_pred_labels.pickle", 'wb') as f:
    #     pickle.dump(train_pred_labels_all, f)
    #
    # with open(f"{output_path}/within_test_real_labels.pickle", 'wb') as f:
    #     pickle.dump(test_real_labels_all, f)
    #
    # with open(f"{output_path}/within_test_pred_labels.pickle", 'wb') as f:
    #     pickle.dump(test_pred_labels_all, f)

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