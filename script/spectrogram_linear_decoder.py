import path_setup
from blind_localization.models.contrastive_pipeline import load_signal_labels, preprocess_data
from blind_localization.data.datasets import *
from blind_localization.visualize import *
import matplotlib
import scipy
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def run_pipeline(sessions, config):

    # load source session data and labels
    print("Loading session data and computing spectrogram...")
    channel_features_all, channel_labels_all = preprocess_data(sessions, config)
        
    # within session leave one session out cross validation
    print("Training model on source session...")
    within_accs, across_accs, within_f1, across_f1 = [], [], [], []
    idx_train, idx_test = train_test_split(range(int(len(config['t_starts']))), test_size=0.2, random_state=66)

    for i in range(len(sessions)):
        n_train_trials, n_channels, n_features = channel_features_all[i][idx_train].shape
        
        X_train = channel_features_all[i][idx_train].reshape(-1, n_features)
        y_train = np.hstack([channel_labels_all[i] for _ in range(len(idx_train))])
        
        X_test = channel_features_all[i][idx_test].reshape(-1, n_features)
        y_test = np.hstack([channel_labels_all[i] for _ in range(len(idx_test))])

        model = LogisticRegression(solver='saga', C=0.1)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        print(f"target session {sessions[i]} accuracy: {balanced_accuracy_score(y_test, y_pred_test)}")
        within_accs.append(balanced_accuracy_score(y_test, y_pred_test))
        within_f1.append(f1_score(y_test, y_pred_test, average='weighted'))

    print("within session accuracy: ", sum(within_accs)/len(within_accs))
    print("within session f1 score: ", sum(within_f1)/len(within_f1))

    # across session
    for i in range(len(sessions)):
        n_train_trials, n_channels, n_features = channel_features_all[i][idx_train].shape
        X_train = np.concatenate([channel_features_all[j][idx_train].reshape(-1, n_features) for j in range(len(sessions)) if j != i], axis=0)        
        y_train = np.hstack([channel_labels_all[j] for _ in range(n_train_trials) for j in range(len(sessions)) if j != i])
        groups = np.concatenate([np.full(n_train_trials*channel_features_all[j].shape[1], j) for j in range(len(sessions)) if j != i])
        
        X_test = channel_features_all[i][idx_test].reshape(-1, n_features)
        y_test = np.hstack([channel_labels_all[i] for _ in range(len(idx_test))])
        
        model = LogisticRegression(solver='saga', C=0.1, warm_start=True)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        print(f"target session {sessions[i]} accuracy: ", balanced_accuracy_score(y_test, y_pred_test))
        across_accs.append(balanced_accuracy_score(y_test, y_pred_test))
        across_f1.append(f1_score(y_test, y_pred_test, average='weighted'))

    print("across session accuracy: ", sum(across_accs)/len(across_accs))
    print("across session f1 score: ", sum(across_f1)/len(across_f1))

    if config["visualize"]: visualize_session_accuracy(within_accs, across_accs, sessions, labels=['within', 'across'])
    return sum(within_accs)/len(within_accs), sum(across_accs)/len(across_accs)

    
def trials_number_experiment(session_names, config):
    end_times = np.arange(2160.5, 2190, 4)
    cv_accuracy_scores = [] 

    for t in end_times:
        config['t_starts'] = np.arange(2160, t, 0.5)
        _, cv_accuracy_score = run_pipeline(session_names, config)
        cv_accuracy_scores.append(cv_accuracy_score)

    n_trials = (end_times - 2160) / 0.5
    visualize_acc_with_data_size(n_trials, cv_accuracy_scores)

if __name__ == "__main__":
    folder_path = "/scratch/th3129/region_decoding/data/Neuronexus"
    config = {
        "visualize": True,
        "one_vs_one": False,
        'swr_flag': False,
        'raw_signal':False,
        'trial_average': True,
        'within_session': False,
        'kernel': 'rbf',
        't_starts': np.arange(2160, 2340, 3),
        'sampling_rate': 20000,
        'trial_length': 3,
        'n_freq_bins': 500,
        'folder_path': folder_path
    }

    # run pipeline for any pairs of source and target session
    # matplotlib.use('TkAgg')
    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1","AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    run_pipeline(session_names, config)

    # trials_number_experiment(session_names, config)
