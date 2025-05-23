import path_setup
import matplotlib
from sklearn.model_selection import train_test_split
from blind_localization.data.datasets import *
from blind_localization.models.contrastive import *
from blind_localization.models.contrastive_pipeline import *
from blind_localization.models.decoder import *
from blind_localization.visualize import *
from tqdm import tqdm
import wandb
import pickle


def preprocess_data(sessions, session_config):
    channel_features_all, channel_labels_all = [], []

    for source_session in sessions:
        T = session_config['t_starts'][-1] - session_config['t_starts'][0] + (session_config['t_starts'][1] -  session_config['t_starts'][0])
        print(f"Session length: {T} sec")
        
        raw_signal, channel_labels, swr_timestamp = load_signal_labels(source_session, session_config['sampling_rate'], T)
        channel_features_ls = np.zeros((len(session_config['t_starts']), np.sum(channel_labels != 5), int(session_config['sampling_rate']*session_config['trial_length'])))

        for i, t_start in enumerate(session_config['t_starts']):
            channel_features = compute_trial(raw_signal, sr=session_config['sampling_rate'], dT=session_config['trial_length'], ts=t_start)
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
    accuracy_scores, f1_scores = [], []
    
    for i in range(len(sessions)):
        group = "within_session" if session_config["within_session"] else "across_session"
        data_size = len(session_config["t_starts"])
        run = wandb.init(project="spectrogram_mlp_decoder", name=f"{sessions[i]}_{group}_decoder_run",group=group, config=sweep_config, reinit=True)

        if session_config['within_session']:
            train_dataloader, val_dataloader, test_dataloader = build_single_session_dataloader(channel_features_all, channel_labels_all, indices, session_config, sweep_config, RawDataset, session_idx=i)
        else:
            train_dataloader, val_dataloader, test_dataloader = build_multi_session_dataloader(channel_features_all, channel_labels_all, indices, sessions, session_config, sweep_config, RawDataset, test_session_idx=i)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_size = train_dataloader.dataset[0][0][0].size()[0]
        model = ContrastiveLearningWithMLP(encoder=None, input_dim=input_size, hidden_dim=sweep_config['decoder_layer_size'], output_dim=5).to(device)

        contrastive_criterion = InfoNCELoss(temperature=sweep_config['temperature'], device=device)
        supervised_criterion = nn.CrossEntropyLoss()

        combined_optimizer = optim.Adam(model.parameters(), lr=sweep_config["decoder_learning_rate"])
        best_decoder_acc = 0

        # train MLP decoder
        for epoch in range(sweep_config['decoder_epochs']):
            decoder_train_metrics = train_decoder(model, train_dataloader, combined_optimizer, contrastive_criterion, supervised_criterion, device=device)
            decoder_val_metrics = validate_decoder(model, val_dataloader, supervised_criterion, device=device)

            if decoder_val_metrics['accuracy'] > best_decoder_acc:
                best_decoder_acc, patience_counter = decoder_val_metrics['accuracy'], 0
            else:
                patience_counter += 1
                if patience_counter >= sweep_config['patience']: break

            print(f'Epoch [{epoch+1}/{sweep_config["decoder_epochs"]}], Train Loss: {decoder_train_metrics["loss"]:.4f}, Val Loss: {decoder_val_metrics["loss"]:.4f}, Train Acc: {decoder_train_metrics["accuracy"]:.4f}, Val Acc: {decoder_val_metrics["accuracy"]:.4f}')
            wandb.log({"decoder_train_loss": decoder_train_metrics["loss"], "decoder_val_loss": decoder_val_metrics['loss'], 
                    "decoder_train_acc": decoder_train_metrics["accuracy"], "decoder_val_acc": decoder_val_metrics['accuracy'], "decoder_epochs": epoch+1})

        # if config["visualize"]: visualize_train_losses(training_losses, validation_losses, labels=['train', 'validation'])
        file_path = os.path.join(session_config['pickle_path'], f'{session_names[i]}_mlp_results.pkl')
        decoder_test_metrics = validate_decoder(model, test_dataloader, supervised_criterion, device=device, pickle_path=file_path)
        print(f"target session {sessions[i]} test accuracy: ", decoder_test_metrics["accuracy"],
              f"target session {sessions[i]} test balanced accuracy: ", decoder_test_metrics["balanced_accuracy"],
              f"target session {sessions[i]} test f1 score: ", decoder_test_metrics["macro_f1"])
        
        wandb.log(decoder_test_metrics)
        accuracy_scores.append(decoder_test_metrics["balanced_accuracy"])
        f1_scores.append(decoder_test_metrics['macro_f1'])

        run.finish()

    return accuracy_scores, f1_scores


if __name__ == "__main__":
    wandb_config = {
        'fc_layer_size': 512,
        'decoder_layer_size': 128,
        'decoder_epochs': 10,
        'patience': 10,
        'latent_size': 128,
        'spectrogram_size': 500,
        'decoder_learning_rate': 2e-5,
        'batch_size': 64,
        'temperature': 0.606
    }

    folder_path = "/scratch/th3129/region_decoding/results"
    session_config = {
        "separate": True,
        "visualize": True,
        'swr_flag': False,
        't_starts': np.arange(2160, 2340, 3),
        'within_session': True,
        'supervised_contrastive': False,
        'sampling_rate': 20000,
        'trial_length': 3,
        'pickle_path': folder_path
    }

    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1","AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    within_accs, within_f1 = run_pipeline(session_names, session_config, wandb_config)

    session_config['within_session'] = False
    across_accs, across_f1 = run_pipeline(session_names, session_config, wandb_config)
    results_path = '/scratch/th3129/region_decoding/results/spectrogram_mlp_decoder.csv'
    
    within_accs, across_accs = load_results(results_path, field='balanced_accuracy')
    within_f1s, across_f1s = load_results(results_path, field='macro_f1')
    visualize_session_accuracy(within_accs, across_accs, session_names, labels=['within', 'across'])

    print(np.mean(within_accs), np.std(within_accs), np.mean(within_f1s), np.std(within_f1s))
    print(np.mean(across_accs), np.std(across_accs), np.mean(across_f1s), np.std(across_f1s))