import matplotlib
import scipy
from sklearn.model_selection import train_test_split
from blind_localization.data.datasets import *
from blind_localization.data.data_loading import *
from blind_localization.data.preprocess import *
from blind_localization.models.contrastive import *
from sklearn import svm
from scipy.signal import spectrogram
from sklearn.metrics import accuracy_score
import wandb
import random
import pickle
from sklearn.model_selection import train_test_split
# from blind_localization.models.contrastive import *

def load_signal_labels(source_session, sample_rate=20000, T=30):
    source_file = load_session_data("script/Neuronexus_preprocessing/file_path_hpc.json", source_session)
    public_file = load_session_data("script/Neuronexus_preprocessing/file_path_hpc.json", "public")

    raw_signal, df, skipped_channels = load_data(source_file["raw_signal_path"], public_file["label_path"], source_file["xml_path"], sheet_name=source_file["sheet_name"],
                                                 sample_rate=sample_rate, T=T)
    channel_region_map, skipped_channels, channel_channel_map = process_labels(df, public_file["mapping_path"], skipped_channels)
    raw_signal = process_signals(raw_signal, channel_channel_map)

    channel_labels = np.load(f"data/Neuronexus/labels_{source_session}.npy")
    labels = np.argmax(channel_labels, axis=1) + 1
    channel_labels = channel_labels * labels[:, np.newaxis]
    class_labels = np.sum(channel_labels, axis=1).astype(int)
    
    dict = {0:5, 1:0, 2:1, 3:2, 4:3, 5:4}
    class_labels = np.array([dict[c] for c in class_labels])

    swr_file = scipy.io.loadmat(source_file["swr_path"])
    swr_timestamp = swr_file['ripples']['timestamps'][0][0]
  
    return raw_signal, class_labels, swr_timestamp

def build_single_session_dataloader(channel_features_all, channel_labels_all, channel_trials_all,
                                    config, Dataset, sessions, session_idx=0, vit=False):
    features = channel_features_all.get(sessions[session_idx])
    labels = channel_labels_all.get(sessions[session_idx])
    trials = channel_trials_all.get(sessions[session_idx])

    # features = [f if f.shape[0] == 3749 else f[:3749] for f in features]
    features = np.array(features)

    trial_length = 60
    train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)
    train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=0.25, random_state=42)

    train_idx = [idx for idx, val in enumerate(trials) if val in train_tr_idx]
    test_idx = [idx for idx, val in enumerate(trials) if val not in train_tr_idx]
    val_idx = [idx for idx, val in enumerate(trials) if val in val_tr_idx]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    train_dataset = Dataset(np.array(X_train), y_train, spectrogram_size=500,
                            time_bins=config["time_bins"], library=config["library"], vit=vit)
    val_dataset = Dataset(np.array(X_val), y_val, spectrogram_size=500,
                          time_bins=config["time_bins"], library=config["library"], vit=vit)
    test_dataset = Dataset(np.array(X_test), y_test, spectrogram_size=500,
                           time_bins=config["time_bins"], library=config["library"], vit=vit)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def build_multi_session_dataloader(channel_features_all, channel_labels_all, channel_trials_all,
                                   sessions, config, Dataset, test_session_idx=0, vit=False):
    X_test = channel_features_all.get(sessions[test_session_idx])
    X_test = [x if x.shape[0] == 3749 else x[:3749] for x in X_test]
    X_test = np.array(X_test)
    y_test = channel_labels_all.get(sessions[test_session_idx])

    trials = channel_trials_all.get(sessions[test_session_idx])

    trial_length = 60
    train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)
    train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=0.25, random_state=42)
    test_idx = [idx for idx, val in enumerate(trials) if val in test_tr_idx]
    X_test = [X_test[idx] for idx in test_idx]
    y_test = [y_test[idx] for idx in test_idx]

    X_train, y_train, X_val, y_val = [], [], [], []
    sessions_except_test_session = [sess for sess in sessions if sess != sessions[test_session_idx]]
    sessions_except_test_session = random.sample(sessions_except_test_session, 4)
    for sess in sessions_except_test_session:
        features = channel_features_all.get(sess)
        features = [f if f.shape[0] == 3749 else f[:3749] for f in features]
        features = np.array(features)
        labels = channel_labels_all.get(sess)
        trial_idx = channel_trials_all.get(sess)

        idx_train = [idx for idx, val in enumerate(trial_idx) if val in train_tr_idx]
        idx_val = [idx for idx, val in enumerate(trial_idx) if val in val_tr_idx]

        X_train.extend(features[idx_train])
        y_train.extend(labels[idx_train])

        X_val.extend(features[idx_val])
        y_val.extend(labels[idx_val])
    
    print(np.array(X_train).shape, np.array(y_train).shape, np.array(X_test).shape, np.array(y_test).shape)

    train_dataset = Dataset(np.array(X_train), y_train, spectrogram_size=500,
                            time_bins=config["time_bins"], library=config["library"], vit=vit)
    val_dataset = Dataset(np.array(X_val), y_val, spectrogram_size=500,
                          time_bins=config["time_bins"], library=config["library"], vit=vit)
    test_dataset = Dataset(np.array(X_test), y_test, spectrogram_size=500,
                           time_bins=config["time_bins"], library=config["library"], vit=vit)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

def build_multi_session_dataloader_new(channel_features_all, channel_labels_all, channel_trials_all,channel_chans_all,
                                   sessions, config, Dataset, test_session_idx=0, vit=False):
    X_test = channel_features_all.get(sessions[test_session_idx])
    # X_test = np.array([x if x.shape[0] == 3749 else x[:3749] for x in X_test])
    y_test = channel_labels_all.get(sessions[test_session_idx])
    trials = channel_trials_all.get(sessions[test_session_idx])
    chans = channel_chans_all.get(sessions[test_session_idx])

    trial_length = 60
    train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)
    train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=0.25, random_state=42)
    test_idx = [idx for idx, val in enumerate(trials) if val in test_tr_idx]
    ttest_idx = [idx for idx, val in enumerate(trials) if val in test_tr_idx]

    X_test = [X_test[idx] for idx in test_idx]
    y_test = [y_test[idx] for idx in test_idx]
    trials_test = [trials[idx] for idx in test_idx]
    chans_test = [chans[idx] for idx in test_idx]
    test_metadata = [f"{sessions[test_session_idx]}_{trials_test[i]}_{chans_test[i]}" for i in range(len(trials_test))]
    assert len(X_test) == len(y_test) == len(trials_test) == len(chans_test)

    train_metadata, val_metadata = [], []

    X_train, y_train, X_val, y_val = [], [], [], []
    available_sessions = [sess for sess in sessions if sess != sessions[test_session_idx]]
    random.seed(42)
    train_sessions = available_sessions.copy()
    val_sessions = [train_sessions[-1]]

    for sess in train_sessions:
        features = channel_features_all.get(sess)
        # features = np.array([f if f.shape[0] == 3749 else f[:3749] for f in features])
        labels = channel_labels_all.get(sess)
        trials = channel_trials_all.get(sess)
        chans = channel_chans_all.get(sess)

        idx_train = [idx for idx, val in enumerate(trials) if val in train_tr_idx]

        X_train.extend(features[idx_train])
        y_train.extend(labels[idx_train])

        trials_train = [trials[idx] for idx in idx_train]
        chans_train = [chans[idx] for idx in idx_train]
        train_metadata.extend([f"{sess}_{trials_train[i]}_{chans_train[i]}" for i in range(len(trials_train))])

    for sess in val_sessions:
        features = channel_features_all.get(sess)
        # features = np.array([f if f.shape[0] == 3749 else f[:3749] for f in features])
        labels = channel_labels_all.get(sess)
        trials = channel_trials_all.get(sess)
        chans = channel_chans_all.get(sess)

        idx_val = [idx for idx, val in enumerate(trials) if val in val_tr_idx]

        X_val.extend(features[idx_val])
        y_val.extend(labels[idx_val])

        trials_val = [trials[idx] for idx in idx_val]
        chans_val = [chans[idx] for idx in idx_val]

        val_metadata.extend([f"{sess}_{trials_val[i]}_{chans_val[i]}" for i in range(len(trials_val))])
    
    print(f"Training on {train_sessions}, Validate on {val_sessions}, Test on {sessions[test_session_idx]}")
    print(f"Training shape: {np.array(X_train).shape}, Validation shpae: {np.array(X_val).shape}, Test shape: {np.array(X_test).shape}")

    train_dataset = Dataset(np.array(X_train), y_train, spectrogram_size=500,
                            time_bins=config["time_bins"], library=config["library"], vit=vit)
    val_dataset = Dataset(np.array(X_val), y_val, spectrogram_size=500,
                          time_bins=config["time_bins"], library=config["library"], vit=vit)
    test_dataset = Dataset(np.array(X_test), y_test, spectrogram_size=500,
                           time_bins=config["time_bins"], library=config["library"], vit=vit)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, train_metadata, val_metadata, test_metadata


def extract_features(model, dataloader, device='cpu', feature_flag=True):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            signal = data[0].to(device)
            label = label.to(device)
            labels.append(label)
            feature = model(signal) if feature_flag else signal
            features.append(feature)

    features = torch.cat(features, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    return features, labels


def train_clf_with_embeddings(model, dataloader, device, subset_size=2000):
    X_embed, y_embed = extract_features(model, dataloader, device)
    if subset_size:
        random_indices = np.random.choice(X_embed.shape[0], subset_size, replace=False)
        X_embed, y_embed = X_embed[random_indices], y_embed[random_indices]
    clf = svm.SVC(C=10, gamma='scale')
    clf.fit(X_embed, y_embed)
    return clf


def eval_clf_with_embeddings(model, dataloader, device, clf):
    X_embed, y_embed = extract_features(model, dataloader, device)
    y_pred = clf.predict(X_embed)
    return y_pred, accuracy_score(y_pred, y_embed)


def train(model, dataloader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = []

    for data, label in dataloader:
        spectrograms, augmented_spectrograms = data
        spectrograms: torch.Tensor = spectrograms
        augmented_spectrograms: torch.Tensor = augmented_spectrograms
        label: torch.Tensor = label

        spectrograms = spectrograms.to(device)
        augmented_spectrograms = augmented_spectrograms.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        z_i = model(spectrograms)
        z_j = model(augmented_spectrograms)
        loss: torch.Tensor = criterion(z_i, z_j)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    
    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss


def validation(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_loss = []

    with torch.no_grad():
        for data, label in dataloader:
            spectrograms, augmented_spectrograms = data
            spectrograms = spectrograms.to(device)
            augmented_spectrograms = augmented_spectrograms.to(device)
            label = label.to(device)

            z_i = model(spectrograms)
            z_j = model(augmented_spectrograms)
            loss = criterion(z_i, z_j)
            total_loss.append(loss.item())
    
    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss

def set_return_session_trial(train_dataloader, val_dataloader, test_dataloader, flag):
    train_dataloader.dataset.return_session_trial = flag
    val_dataloader.dataset.return_session_trial = flag
    test_dataloader.dataset.return_session_trial = flag