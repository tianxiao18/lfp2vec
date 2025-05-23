import path_setup
import matplotlib
from sklearn.model_selection import train_test_split
from blind_localization.data.datasets import *
from blind_localization.models.contrastive import *
from blind_localization.models.contrastive_pipeline import *
from blind_localization.visualize import *
from sklearn.metrics import accuracy_score
import wandb
import pickle


def preprocess_data(sessions, session_config):
    channel_features_all, channel_labels_all = [], []

    for source_session in sessions:
        T = session_config['t_starts'][-1] - session_config['t_starts'][0] + (
                    session_config['t_starts'][1] - session_config['t_starts'][0])
        print(f"Session length: {T} sec")

        raw_signal, channel_labels, swr_timestamp = load_signal_labels(source_session, session_config['sampling_rate'],
                                                                       T)
        channel_features_ls = np.zeros((len(session_config['t_starts']), np.sum(channel_labels != 5),
                                        int(session_config['sampling_rate'] * session_config['trial_length'])))

        for i, t_start in enumerate(session_config['t_starts']):
            channel_features = compute_trial(raw_signal, sr=session_config['sampling_rate'],
                                             dT=session_config['trial_length'], ts=t_start)
            channel_features_ls[i] = channel_features[channel_labels != 5]

        channel_features_all.append(channel_features_ls)
        channel_labels_all.append(channel_labels[channel_labels != 5])

    return channel_features_all, channel_labels_all


def run_pipeline(sessions, session_config, sweep_config):
    # load source session data and labels
    print("Preprocessing data...")
    channel_features_all, channel_labels_all = preprocess_data(sessions, session_config)

    print("Training model on source session...")
    idx_train, idx_test = train_test_split(range(len(session_config['t_starts'])), test_size=0.2, random_state=66)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.25, random_state=66)
    indices = idx_train, idx_val, idx_test
    accuracy_scores = []

    for i in range(len(sessions)):
        group = "within_session" if session_config["within_session"] else "across_session"
        run = wandb.init(project="scratch", name=f"{sessions[i]}_{group}_supervised_run", group=group,
                         config=sweep_config, reinit=True)

        if session_config['within_session']:
            train_dataloader, val_dataloader, test_dataloader = build_single_session_dataloader(channel_features_all,
                                                                                                channel_labels_all,
                                                                                                indices, session_config,
                                                                                                sweep_config,
                                                                                                RawDataset,
                                                                                                session_idx=i)
        else:
            train_dataloader, val_dataloader, test_dataloader = build_multi_session_dataloader(channel_features_all,
                                                                                               channel_labels_all,
                                                                                               indices, sessions,
                                                                                               session_config,
                                                                                               sweep_config, RawDataset,
                                                                                               test_session_idx=i)

        device = torch.device("cuda")
        input_size = train_dataloader.dataset[0][0][0].size()[0]
        model = ContrastiveEncoder(sweep_config['fc_layer_size'], input_size=input_size,
                                   output_size=sweep_config['latent_size']).to(device)
        criterion = InfoNCELoss(temperature=sweep_config['temperature'], device=device)
        optimizer = optim.Adam(model.parameters(), lr=sweep_config["learning_rate"])

        training_losses, validation_losses = [], []
        best_val_acc, patience_counter = 0, 0

        for epoch in range(sweep_config["epochs"]):
            train_loss = train(model, train_dataloader, optimizer, criterion, device)
            val_loss = validation(model, val_dataloader, criterion, device)

            clf = train_clf_with_embeddings(model, train_dataloader, device,
                                            subset_size=2000)  # this is a lower bound of accuracy (diff <0.05)
            _, train_acc = eval_clf_with_embeddings(model, train_dataloader, device, clf)
            _, val_acc = eval_clf_with_embeddings(model, val_dataloader, device, clf)

            print(
                f'Epoch [{epoch + 1}/{sweep_config["epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train acc: {train_acc: .4f}, Val acc: {val_acc:.4f}')
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc,
                       "epoch": epoch + 1})

            training_losses.append(train_loss)
            validation_losses.append(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= sweep_config['patience']:
                    break

        # if config["visualize"]: visualize_train_losses(training_losses, validation_losses, labels=['train', 'validation'])
        clf = train_clf_with_embeddings(model, train_dataloader, device)
        _, test_acc = eval_clf_with_embeddings(model, test_dataloader, device, clf)
        print(f"target session {sessions[i]} test accuracy: ", test_acc)
        wandb.log({"test_acc": test_acc})
        accuracy_scores.append(test_acc)
        run.finish()

    return accuracy_scores


if __name__ == "__main__":
    folder_path = "/scratch/th3129/region_decoding/data/Neuronexus"

    wandb_config = {
        'fc_layer_size': 512,
        'epochs': 100,
        'patience': 10,
        'latent_size': 8,
        'spectrogram_size': 500,
        'learning_rate': 4.6e-5,
        'batch_size': 184,
        'temperature': 0.69
    }

    session_config = {
        "visualize": True,
        'swr_flag': False,
        't_starts': np.arange(2160, 2340, 3),
        'within_session': True,
        'supervised_contrastive': False,
        'sampling_rate': 20000,
        'trial_length': 3,
        'folder_path': folder_path
    }

    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    within_accs = run_pipeline(session_names, session_config, wandb_config)

    session_config['within_session'] = False
    across_accs = run_pipeline(session_names, session_config, wandb_config)
    # within_accs, across_accs = load_results('/scratch/th3129/region_decoding/results/svm_decoder_3s_summary.csv')
    visualize_session_accuracy(within_accs, across_accs, session_names, labels=['within', 'across'])

    print(within_accs, across_accs)