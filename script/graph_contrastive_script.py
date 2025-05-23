import path_setup
import matplotlib
from sklearn.model_selection import train_test_split
from blind_localization.data.graph_datasets import *
from blind_localization.data.graph_preprocess import *

from blind_localization.models.contrastive import *
from blind_localization.models.contrastive_pipeline import *
from blind_localization.models.encoder import *
from blind_localization.models.decoder import *
from blind_localization.models.graph_training import *

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
    accuracy_scores = []
    
    i = 0
    group = "within_session" if session_config["within_session"] else "across_session"
    data_size = len(session_config["t_starts"])
    run = wandb.init(name=f"{sessions[i]}_{group}_decoder_run",group=group, config=sweep_config, reinit=True)
    sweep_config = wandb.config

    if session_config['within_session']:
        train_dataloader, val_dataloader, test_dataloader = build_single_session_graph_dataloader(channel_features_all, channel_labels_all, indices, session_config, sweep_config, GraphDataset, session_idx=i)
    else:
        train_dataloader, val_dataloader, test_dataloader = build_multi_session_graph_dataloader(channel_features_all, channel_labels_all, indices, sessions, session_config, sweep_config, GraphDataset, test_session_idx=i)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = train_dataloader.dataset[0].num_node_features
    encoder = GraphEncoder(input_size, sweep_config['latent_size'], k=sweep_config['n_layers'], tau=sweep_config['temperature']).to(device)
    model = GraphDecoder(encoder, input_dim=sweep_config['latent_size'], hidden_dim=sweep_config['decoder_layer_size'], output_dim=5).to(device)

    contrastive_criterion = InfoNCELoss(temperature=sweep_config['temperature'], device=device)
    supervised_criterion = nn.CrossEntropyLoss()

    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=sweep_config["encoder_learning_rate"])
    combined_optimizer = optim.Adam(model.parameters(), lr=sweep_config["decoder_learning_rate"])
    best_decoder_acc = 0

    # initial pre-training of encoder
    for epoch in range(sweep_config["encoder_epochs"]):
        train_loss = train_graph_encoder(model.encoder, train_dataloader, encoder_optimizer, device, drop_edge_rate=0.1)
        val_loss = validate_graph_encoder(model.encoder, val_dataloader, device)
        
        print(f'Epoch [{epoch+1}/{sweep_config["encoder_epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        wandb.log({"train_loss": train_loss, "val_loss": val_loss,  "encoder_epochs": epoch+1})

    # fine tuning encoder + decoder OR freeze encoder + train decoder
    for epoch in range(sweep_config['decoder_epochs']):
        decoder_train_loss, decoder_train_acc = train_graph_decoder(model, train_dataloader, combined_optimizer, supervised_criterion, separate=session_config["separate"], device=device)
        decoder_val_loss, decoder_val_acc = validate_graph_decoder(model, val_dataloader, supervised_criterion, device=device)

        if decoder_val_acc > best_decoder_acc:
            best_decoder_acc, patience_counter = decoder_val_acc, 0
        else:
            patience_counter += 1
            if patience_counter >= sweep_config['patience']: break

        print(f'Epoch [{epoch+1}/{sweep_config["decoder_epochs"]}], Train Loss: {decoder_train_loss:.4f}, Val Loss: {decoder_val_loss:.4f}, Train Acc: {decoder_train_acc:.4f}, Val Acc: {decoder_val_acc:.4f}')
        wandb.log({"decoder_train_loss": decoder_train_loss, "decoder_val_loss": decoder_val_loss, 
                "decoder_train_acc": decoder_train_acc, "decoder_val_acc": decoder_val_acc, "decoder_epochs": epoch+1})

    # if config["visualize"]: visualize_train_losses(training_losses, validation_losses, labels=['train', 'validation'])
    _, decoder_test_acc = validate_graph_decoder(model, test_dataloader, supervised_criterion, device=device)
    print(f"target session {sessions[i]} test accuracy: ", decoder_test_acc)
    wandb.log({"n_trials":data_size, "test_acc": decoder_test_acc})
    accuracy_scores.append(decoder_test_acc)

    run.finish()

    return accuracy_scores


def run_pipeline_wrapper():
    session_config = {
        "separate": True,
        "visualize": True,
        'swr_flag': False,
        't_starts': np.arange(2160, 2340, 3),
        'within_session': False,
        'supervised_contrastive': False,
        'sampling_rate': 20000,
        'trial_length': 3
    }

    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1","AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    test_acc = run_pipeline(session_names, session_config, wandb.config)
    return test_acc


if __name__ == "__main__":
    sweep_config = {'method': 'random'}
    parameters_dict = {
        'n_layers': {'values': [2, 3, 4, 5]},
        'fc_layer_size': {'values': [128, 256, 512]},
        'decoder_layer_size': {'values': [128, 256, 512]},
        'encoder_epochs': {'values': [50]},
        'decoder_epochs': {'values': [10, 20, 50]},
        'patience': {'values':[5, 10, 15]},
        'latent_size': {'values': [4, 8, 16, 32, 64, 128, 256, 512]},
        'spectrogram_size': {'values': [500]},
        'encoder_learning_rate': {'distribution': 'uniform', 'min':1e-5, 'max':1e-4},
        'decoder_learning_rate': {'distribution': 'uniform', 'min':1e-5, 'max':1e-4},
        'batch_size': {'distribution':'q_log_uniform_values','q': 8,'min': 32,'max': 256},
        'temperature': {'distribution': 'uniform', 'min':0.1, 'max':0.9}
    }
    sweep_config['parameters'] = parameters_dict
    
    sweep_id = wandb.sweep(sweep_config, project="graph_decoder_sweep")
    wandb.agent(sweep_id, run_pipeline_wrapper, count=5)

    # wandb_config = {
    #     'n_layers': 2,
    #     'decoder_layer_size': 512,
    #     'encoder_epochs': 20,
    #     'decoder_epochs': 10,
    #     'patience': 10,
    #     'latent_size': 256,
    #     'spectrogram_size': 500,
    #     'encoder_learning_rate': 5.7e-5,
    #     'decoder_learning_rate': 9.7e-5,
    #     'batch_size': 224,
    #     'temperature': 0.27
    # }

    # session_config = {
    #     "separate": True,
    #     "visualize": True,
    #     'swr_flag': False,
    #     't_starts': np.arange(2160, 2340, 3),
    #     'within_session': False,
    #     'supervised_contrastive': False,
    #     'sampling_rate': 20000,
    #     'trial_length': 3
    # }

    # session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1","AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    # within_accs = run_pipeline(session_names, session_config, wandb_config)

    # session_config['within_session'] = False
    # across_accs = run_pipeline(session_names, session_config, wandb_config)
    # within_accs, across_accs = load_results('/scratch/th3129/region_decoding/results/mlp_decoder_results.csv')
    # visualize_session_accuracy(within_accs, across_accs, session_names, labels=['within', 'across'])

    # print(within_accs, across_accs)