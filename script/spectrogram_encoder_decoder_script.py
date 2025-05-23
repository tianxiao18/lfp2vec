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
from script.utils import save_if_better


def load_preprocessed_data(pickle_path, file_path):
    features, labels, trials = {}, {}, {}
    for session in file_path:
        data = pickle.load(open(f"{pickle_path}/{session}_raw.pickle", 'rb'))
        # data = pickle.load(open(f"{pickle_path}/{session}_lfp.pickle", 'rb'))
        X, y, trial_idx = zip(*[(d[0], d[1], d[2]) for d in data])
        features[session] = np.array(X)
        labels[session] = np.array(y, dtype=int)
        trials[session] = np.array(trial_idx)

        non_zero_indices = [i for i, x in enumerate(features[session]) if not np.all(x == 0)]

        features[session] = features[session][non_zero_indices]
        labels[session] = labels[session][non_zero_indices]
        trials[session] = trials[session][non_zero_indices]

        # Sanity check
        assert len(features[session]) == len(labels[session]) == len(trials[session]), \
            f"Inconsistent data sizes for session {session}"

    return features, labels, trials

def run_pipeline(sessions, session_config, sweep_config, i):

    # load source session data and labels
    print("Preprocessing data...")
    channel_features_all, channel_labels_all, channel_trials_all = load_preprocessed_data(pickle_path, sessions)

    print("Training model on source session...")
    idx_train, idx_test = train_test_split(range(len(session_config['t_starts'])), test_size=0.2, random_state=66)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.25, random_state=66)
    indices = idx_train, idx_val, idx_test
    accuracy_scores, f1_scores = [], []
    
    group = "within_session" if session_config["within_session"] else "across_session"
    data_size = len(session_config["t_starts"])
    run = wandb.init(project="scratch", name=f"{sessions[i]}_{group}_decoder_run",group=group, config=sweep_config, reinit=True)
    # sweep_config = wandb.config

    if session_config['within_session']:
        train_dataloader, val_dataloader, test_dataloader = build_single_session_dataloader(channel_features_all,
                                                                                            channel_labels_all,
                                                                                            channel_trials_all,
                                                                                            sweep_config,
                                                                                            RawDataset, sessions,
                                                                                            session_idx=i)
    else:
        train_dataloader, val_dataloader, test_dataloader = build_multi_session_dataloader(channel_features_all,
                                                                                            channel_labels_all,
                                                                                            channel_trials_all,
                                                                                            sessions,
                                                                                            sweep_config, RawDataset,
                                                                                            test_session_idx=i)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = train_dataloader.dataset[0][0][0].size()[0]
    encoder = ContrastiveEncoder(sweep_config['fc_layer_size'], input_size=input_size, output_size=sweep_config['latent_size']).to(device)
    model = ContrastiveLearningWithMLP(encoder, input_dim=sweep_config['latent_size'], hidden_dim=sweep_config['decoder_layer_size'], output_dim=5).to(device)

    contrastive_criterion = InfoNCELoss(temperature=sweep_config['temperature'], device=device)
    supervised_criterion = nn.CrossEntropyLoss()

    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=sweep_config["encoder_learning_rate"])
    combined_optimizer = optim.Adam(model.parameters(), lr=sweep_config["decoder_learning_rate"])
    best_decoder_loss = 10000
    patience_counter = 0
    
    # initial pre-training of encoder
    for epoch in range(sweep_config["encoder_epochs"]):
        train_loss = train(model.encoder, train_dataloader, encoder_optimizer, contrastive_criterion, device)
        val_loss = validation(model.encoder, val_dataloader, contrastive_criterion, device)
        
        print(f'Epoch [{epoch+1}/{sweep_config["encoder_epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        wandb.log({"train_loss": train_loss, "val_loss": val_loss,  "encoder_epochs": epoch+1})

    # fine tuning encoder + decoder OR freeze encoder + train decoder
    for epoch in range(sweep_config['decoder_epochs']):
        decoder_train_metrics = train_decoder(model, train_dataloader, combined_optimizer, contrastive_criterion, supervised_criterion, mode=session_config['separate'], device=device)
        decoder_val_metrics = validate_decoder(model, val_dataloader, supervised_criterion, device=device)

        if decoder_val_metrics['loss'] < best_decoder_loss:
            best_decoder_loss, patience_counter = decoder_val_metrics['loss'], 0
        else:
            patience_counter += 1
            if patience_counter >= sweep_config['patience']: break

        print(f'Epoch [{epoch+1}/{sweep_config["decoder_epochs"]}], Train Loss: {decoder_train_metrics["loss"]:.4f}, Val Loss: {decoder_val_metrics["loss"]:.4f}, Train Acc: {decoder_train_metrics["balanced_accuracy"]:.4f}, Val Acc: {decoder_val_metrics["balanced_accuracy"]:.4f}')
        wandb.log({"decoder_train_loss": decoder_train_metrics["loss"], "decoder_val_loss": decoder_val_metrics['loss'], 
                "decoder_train_acc": decoder_train_metrics["balanced_accuracy"], "decoder_val_acc": decoder_val_metrics['balanced_accuracy'], "decoder_epochs": epoch+1})

    # if config["visualize"]: visualize_train_losses(training_losses, validation_losses, labels=['train', 'validation'])
    decoder_test_metrics = validate_decoder(model, test_dataloader, supervised_criterion, device=device)
    print(f"target session {sessions[i]} test balanced accuracy: ", decoder_test_metrics["balanced_accuracy"],
        f"target session {sessions[i]} test f1 score: ", decoder_test_metrics["macro_f1"])

    set_return_session_trial(train_dataloader, val_dataloader, test_dataloader, flag=True)

    wandb.log(decoder_test_metrics)
    accuracy_scores.append(decoder_test_metrics["balanced_accuracy"])
    f1_scores.append(decoder_test_metrics['macro_f1'])
    decoder_test_acc = float(decoder_test_metrics['balanced_accuracy'])
    test_real_labels, test_pred_labels = decoder_test_metrics['real_labels'], decoder_test_metrics['predicted_labels']
    train_real_labels, train_pred_labels = decoder_train_metrics['real_labels'], decoder_train_metrics['predicted_labels']

    test_acc = 0
    if session_config['within_session']:
        sess_type = 'within'
    else:
        sess_type = 'across'

    output_path = f"results/Neuronexus/spectrogram/SimCLR_MLP"

    if session_config['within_session']:
        title = f"{sessions[i]}_within_session"
    else:
        title = f"{sessions[i]}_across_session"

    if os.path.exists(f"{output_path}/{sessions[i]}_{sess_type}_session.pickle"):
        # Load the existing pickle file
        with open(f"{output_path}/{sessions[i]}_{sess_type}_session.pickle", 'rb') as f:
            saved_data = pickle.load(f)
            test_acc = saved_data['test_acc']

    # Save encoder, model and sweep_config
    if decoder_test_acc > test_acc:
        torch.save(encoder.state_dict(), f"{output_path}/{sessions[i]}_{sess_type}_session_model.pth")
        torch.save(model.state_dict(), f"{output_path}/{sessions[i]}_{sess_type}_session_decoder.pth")
        params = ['fc_layer_size', 'decoder_layer_size', 'encoder_epochs', 'decoder_epochs', 'patience', 'latent_size',
                  'spectrogram_size', 'encoder_learning_rate', 'decoder_learning_rate', 'batch_size', 'temperature',
                  'time_bins', 'library', 'dataset']
        sweep_dict = {}
        for key in params:
            if key in sweep_config:
                sweep_dict[key] = sweep_config[key]
        with open(f"{output_path}/{sessions[i]}_{sess_type}_session_sweep_config.pickle", 'wb') as f:
            pickle.dump(sweep_dict, f)

        output_path = f"{output_path}/{title}.pickle"
        save_if_better(output_path, test_acc, train_real_labels, train_pred_labels, test_real_labels, test_pred_labels)


    run.finish()

    return accuracy_scores


def run_pipeline_wrapper(i, is_within, output_path):
    session_config = {
        "separate": True,
        "visualize": True,
        'swr_flag': False,
        't_starts': np.arange(2160, 2340, 3),
        'within_session': is_within,
        'supervised_contrastive': False,
        'sampling_rate': 20000,
        'trial_length': 3,
        'session_idx': True
    }
    output_path = f"{output_path}/{session_config['separate']}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    sessions_list = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]

    run_pipeline(sessions_list, session_config, wandb.config, i)

    # test_acc = round(decoder_test_acc, 5)

    if session_config['within_session']:
        title = f"{sessions_list[i]}_within_session"
    else:
        title = f"{sessions_list[i]}_across_session"

    output_path = f"{output_path}/{title}.pickle"
    # save_if_better(output_path, test_acc, train_real_labels, train_pred_labels, test_real_labels, test_pred_labels)

    return None

def plot_accuracy_comparison():
    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1","AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    
    within_accs_before, across_accs_before = load_results('/scratch/th3129/region_decoding/results/spectrogram_simclr_mlp.csv', field='accuracy')
    within_accs_after, _ = load_results('/scratch/th3129/region_decoding/results/simclr_after_merge.csv', field='test_acc')

    font = {'size': 18}
    matplotlib.rc('font', **font)
    
    colors = ['#FDB462', '#80B1D3']
    visualize_session_metrics([within_accs_before, within_accs_after], session_names, labels=['simclr_before', 'simclr_after'], colors=colors, title='simclr_comparison')

pickle_path = f'/scratch/cl7201/shared/Neuronexus/spectrogram'
# pickle_path = f'/scratch/th3129/region_decoding/data/Neuronexus/lfp'
output_path = f"results/Neuronexus/spectrogram/SimCLR_MLP"

if __name__ == "__main__":
    # plot_accuracy_comparison()
    wandb_config = {
        'fc_layer_size': 512,
        'decoder_layer_size': 128,
        'encoder_epochs': 50,
        'decoder_epochs': 20,
        'patience': 3,
        'latent_size': 128,
        'spectrogram_size': 500,
        'encoder_learning_rate': 8e-5,
        'decoder_learning_rate': 5e-5,
        'batch_size': 64,
        'temperature': 0.606,
        'time_bins': 16,
        'library': 'librosa'
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
        'session_idx': True
    }

    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    for i in range(len(session_names)):
        within_accs = run_pipeline(session_names, session_config, wandb_config, i)

    session_config['within_session'] = False
    for i in range(len(session_names)):
        across_accs = run_pipeline(session_names, session_config, wandb_config, i)
    # results_path = '/scratch/th3129/region_decoding/results/spectrogram_simclr_mlp.csv'

    # within_accs, across_accs = load_results(results_path, field='balanced_accuracy')
    # within_f1s, across_f1s = load_results(results_path, field='macro_f1')
    # visualize_session_accuracy(within_accs, across_accs, session_names, labels=['within', 'across'])

    # print(np.mean(within_accs), np.std(within_accs), np.mean(within_f1s), np.std(within_f1s))
    # print(np.mean(across_accs), np.std(across_accs), np.mean(across_f1s), np.std(across_f1s))

    # is_within = True
    # sweep_config = {'method': 'random'}
    # parameters_dict = {
    #     'fc_layer_size': {'values': [128, 256, 512]},
    #     'decoder_layer_size': {'values': [128, 256, 512]},
    #     'encoder_epochs': {'values': [50]},
    #     'decoder_epochs': {'values': [10, 20, 50]},
    #     'patience': {'values': [5, 10, 15]},
    #     'latent_size': {'values': [4, 8, 16, 32, 64, 128, 256, 512]},
    #     'spectrogram_size': {'values': [500]},
    #     'encoder_learning_rate': {'distribution': 'uniform', 'min': 1e-5, 'max': 1e-4},
    #     'decoder_learning_rate': {'distribution': 'uniform', 'min': 1e-5, 'max': 1e-4},
    #     'batch_size': {'distribution': 'q_log_uniform_values', 'q': 8, 'min': 32, 'max': 256},
    #     'temperature': {'distribution': 'uniform', 'min': 0.1, 'max': 0.9},
    #     'time_bins': {'values': [16]},
    #     'library': {'values': ['librosa']}
    # }
    # sweep_config['parameters'] = parameters_dict

    # sess = "within" if is_within else "across"
    # data = "Neuronexus"

    # sessions_list = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    # sweep_id_json_path = 'sweep_id.json'
    # if not os.path.exists(sweep_id_json_path):
    #     with open(sweep_id_json_path, 'w') as f:
    #         json.dump({}, f)
    # sweep_id_json = json.load(open(sweep_id_json_path, 'r'))

    # for i in range(len(sessions_list)):
    #     run_pipeline_with_args = partial(run_pipeline_wrapper, i, is_within, output_path=None)

    #     if f"{sessions_list[i]}_{is_within}" not in sweep_id_json:
    #         sweep_id = wandb.sweep(sweep_config, project=f"HPC_SimCLR_{sess}_session_{data}")
    #     else:
    #         sweep_id = wandb.sweep(sweep_config, project=f"HPC_SimCLR_{sess}_session_{data}"
    #                                , prior_runs=sweep_id_json[f"{sessions_list[i]}_{is_within}"])

    #     wandb.agent(sweep_id, run_pipeline_with_args, count=10)