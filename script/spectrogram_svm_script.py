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
from sklearn.metrics import accuracy_score


def one_vs_one_pipeline(channel_features_all, channel_labels_all, sessions):
    train_accs = np.zeros((len(sessions), len(sessions)))
    test_accs = np.zeros((len(sessions), len(sessions)))

    for i in range(len(sessions)):
        for j in range(len(sessions)):
            X_source, y_source = channel_features_all[i], channel_labels_all[i]
            X_target, y_target = channel_features_all[j], channel_labels_all[j]

            model = svm.SVC()      
            model.fit(X_source, y_source)

            y_pred_train = model.predict(X_source)     
            train_accs[i][j] = accuracy_score(y_source, y_pred_train)

            y_pred_test = model.predict(X_target)
            test_accs[i][j] = accuracy_score(y_target, y_pred_test)
            print(f"target session {sessions[j]} accuracy: ", test_accs[i][j])
            
    return train_accs, test_accs


def prepare_trial_averaged_data(X, y, n_splits=5):
    """
        X: data of shape (n_trials, n_channels, n_features)
        y: label of shape (n_channels, )
        return (n_channels, n_features), (n_channels, ), (n_trials, )
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    averaged_X, averaged_y, groups = [], [], []
    group_cnt = 0

    for train_index, val_index in kf.split(X):
        X_val = np.mean(X[val_index], axis=0)
        averaged_X.append(X_val)
        averaged_y.append(y)
        groups.extend([group_cnt] * len(X_val))
        group_cnt += 1

    return np.vstack(averaged_X), np.hstack(averaged_y), np.array(groups)



def run_pipeline(sessions, config):

    # matplotlib.use('TkAgg')

    # load source session data and labels
    print("Loading session data and computing spectrogram...")
    channel_features_all, channel_labels_all = preprocess_data(sessions, config)

    # one vs one model evaluation
    if config["one_vs_one"]:
        return one_vs_one_pipeline(channel_features_all, channel_labels_all, sessions)
       
    
    # within session leave one session out cross validation
    print("Training model on source session...")
    within_accs, across_accs = [], []
    idx_train, idx_test = train_test_split(range(len(config['t_starts'])), test_size=0.2, random_state=66)
    
    for i in range(len(sessions)):
        X_train, y_train = channel_features_all[i][idx_train], channel_labels_all[i]
        X_avg, y_avg, groups = prepare_trial_averaged_data(X_train, y_train, n_splits=5)

        X_test = np.mean(channel_features_all[i][idx_test], axis=0)
        y_test = channel_labels_all[i]

        model = svm.SVC()
        
        param_grid = {'C': [0.1, 1, 10], 'gamma':['scale', 'auto']}
        logo = LeaveOneGroupOut()
        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=logo, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_avg, y_avg, groups=groups)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Best Parameters: {best_params}\nBest CV Score: {best_score}")

        model.set_params(**best_params)
        model.fit(np.mean(X_train, axis=0), y_train)
        y_pred_test = model.predict(X_test)
        print(f"target session {sessions[i]} accuracy: ", accuracy_score(y_test, y_pred_test))
        within_accs.append(accuracy_score(y_test, y_pred_test))
    print("within CV accuracy: ", sum(within_accs)/len(within_accs))
    
    # across session leave one session out cross validation
    for i in range(len(sessions)):
        X_train = np.vstack([np.mean(channel_features_all[j][idx_train], axis=0) for j in range(len(sessions)) if j != i])
        y_train = np.hstack(channel_labels_all[:i]+channel_labels_all[i+1:])
        X_test = np.mean(channel_features_all[i][idx_test], axis=0)
        y_test = channel_labels_all[i]
        
        model = svm.SVC()
        param_grid = {'C': [0.1, 1, 10], 'gamma':['scale', 'auto']}
        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Best Parameters: {best_params}\nBest CV Score: {best_score}")

        model.set_params(**best_params)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        print(f"target session {sessions[i]} accuracy: ", accuracy_score(y_test, y_pred_test))
        across_accs.append(accuracy_score(y_test, y_pred_test))
    print("across CV accuracy: ", sum(across_accs)/len(across_accs))

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
    t_starts = np.arange(2160, 2190, 0.5)
    folder_path = "/scratch/th3129/region_decoding/data/Neuronexus"
    
    config = {
        "visualize": True,
        "one_vs_one": False,
        'swr_flag': False,
        'raw_signal':False,
        'trial_average': True,
        't_starts': t_starts,
        'within_session': False,
        't_starts': np.arange(2160, 2340, 3),
        'sampling_rate': 20000,
        'trial_length': 3,
        'n_freq_bins': 500,
        'folder_path': folder_path
    }

    # run pipeline for any pairs of source and target session
    session_names = ["AD_HF01_1"]
    run_pipeline(session_names, config)

    # config['one_vs_one'] = False
    # region_accuracy_scores_new = run_pipeline(session_names, config)
    # visualize_box_plot(list(region_accuracy_scores_new), list(region_accuracy_scores), model_names=["raw_signal", "spectrogram"])

    # trials_number_experiment(session_names, config)
