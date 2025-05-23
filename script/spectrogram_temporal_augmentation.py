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
        run = wandb.init(project="st_augmentation", name=f"{sessions[i]}_{group}_decoder_run",group=group, config=sweep_config, reinit=True)
        # sweep_config = wandb.config

        if session_config['within_session']:
            train_dataloader, val_dataloader, test_dataloader = build_single_session_dataloader(channel_features_all, channel_labels_all, indices, session_config, sweep_config, ImageDataset, session_idx=i)
        else:
            train_dataloader, val_dataloader, test_dataloader = build_multi_session_dataloader(channel_features_all, channel_labels_all, indices, sessions, session_config, sweep_config, ImageDataset, test_session_idx=i)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # input_size = train_dataloader.dataset[0][0][0].size()[0]
        encoder = ResNetEncoder(input_size=sweep_config['latent_size'], output_size=sweep_config['latent_size'], use_projector=True).to(device)
        model = ContrastiveLearningWithMLP(encoder, input_dim=sweep_config['latent_size'], hidden_dim=sweep_config['decoder_layer_size'], output_dim=5).to(device)

        contrastive_criterion = InfoNCELoss(temperature=sweep_config['temperature'], device=device)
        supervised_criterion = nn.CrossEntropyLoss()

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=sweep_config["encoder_learning_rate"])
        combined_optimizer = optim.Adam(model.parameters(), lr=sweep_config["decoder_learning_rate"])
        best_decoder_acc = 0
        
        # initial pre-training of encoder
        for epoch in range(sweep_config["encoder_epochs"]):
            train_loss = train(encoder, train_dataloader, encoder_optimizer, contrastive_criterion, device)
            val_loss = validation(encoder, val_dataloader, contrastive_criterion, device)
            
            print(f'Epoch [{epoch+1}/{sweep_config["encoder_epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            wandb.log({"train_loss": train_loss, "val_loss": val_loss,  "encoder_epochs": epoch+1})
        
        encoder.use_projector = False
        # fine tuning encoder + decoder OR freeze encoder + train decoder
        for epoch in range(sweep_config['decoder_epochs']):
            decoder_train_metrics = train_decoder(model, train_dataloader, combined_optimizer, contrastive_criterion, supervised_criterion, mode=session_config['separate'], device=device)
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
        file_path = os.path.join(session_config['pickle_path'], f'{sessions[i]}_simclr_results.pkl')
        decoder_test_metrics = validate_decoder(model, test_dataloader, supervised_criterion, device=device, pickle_path=file_path)
        print(f"target session {sessions[i]} test balanced accuracy: ", decoder_test_metrics["balanced_accuracy"],
            f"target session {sessions[i]} test f1 score: ", decoder_test_metrics["macro_f1"])

        wandb.log(decoder_test_metrics)
        accuracy_scores.append(decoder_test_metrics["balanced_accuracy"])
        f1_scores.append(decoder_test_metrics['macro_f1'])

        run.finish()

    return accuracy_scores

def run_pipeline_wrapper():
    results_folder_path = "/scratch/th3129/region_decoding/results"
    session_config = {
        "separate": True,
        "visualize": False,
        'swr_flag': False,
        't_starts': np.arange(2160, 2340, 3),
        'within_session': True,
        'supervised_contrastive': False,
        'sampling_rate': 20000,
        'trial_length': 3,
        'pickle_path': results_folder_path,
        'session_idx': False,
        'augmentation_type': "temporal_neighbors"
    }

    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2","NN_syn_01", "NN_syn_02"]
    test_acc = run_pipeline(session_names, session_config, wandb.config)
    return test_acc

if __name__ == "__main__":
    # sweep_config = {'method': 'random'}
    # parameters_dict = {
    #     'fc_layer_size': {'values': [128, 256, 512]},
    #     'decoder_layer_size': {'values': [128, 256, 512]},
    #     'encoder_epochs': {'values': [10, 20, 30]},
    #     'decoder_epochs': {'values': [10, 20]},
    #     'patience': {'values':[5, 10, 15]},
    #     'latent_size': {'values': [4, 8, 16, 32, 64, 128, 256, 512]},
    #     'spectrogram_size': {'values': [500]},
    #     'encoder_learning_rate': {'distribution': 'uniform', 'min':1e-5, 'max':1e-4},
    #     'decoder_learning_rate': {'distribution': 'uniform', 'min':1e-5, 'max':1e-4},
    #     'batch_size': {'distribution':'q_log_uniform_values','q': 8,'min': 32,'max': 256},
    #     'temperature': {'distribution': 'uniform', 'min':0.1, 'max':0.9},
    # }
    # sweep_config['parameters'] = parameters_dict
    
    # sweep_id = wandb.sweep(sweep_config, project="temp_augmentation")
    # wandb.agent(sweep_id, run_pipeline_wrapper, count=10)

    wandb_config = {
        'fc_layer_size': 512,
        'decoder_layer_size': 128,
        'encoder_epochs': 10,
        'decoder_epochs': 20,
        'patience': 10,
        'latent_size': 128,
        'spectrogram_size': 500,
        'encoder_learning_rate': 8e-5,
        'decoder_learning_rate': 5e-5,
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
        'pickle_path': folder_path,
        'session_idx': False,
        'augmentation_type': "spatial_neighbors"
    }

    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2","NN_syn_01", "NN_syn_02"]
    within_accs = run_pipeline(session_names, session_config, wandb_config)

    session_config['within_session'] = False
    across_accs = run_pipeline(session_names, session_config, wandb_config)