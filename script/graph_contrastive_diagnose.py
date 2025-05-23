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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from blind_localization.visualize import *
from tqdm import tqdm
import wandb
import pickle


def preprocess_data(sessions, session_config):
    channel_features_all, channel_labels_all, indices = [], [], []

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
        indices.append(np.arange(1024)[channel_labels != 5])

    return channel_features_all, channel_labels_all, indices

def save_grad_norm_hook(name, grad_norms_per_layer):
    def hook(grad):
        norm = grad.norm(2).item()
        if name not in grad_norms_per_layer:
            grad_norms_per_layer[name] = []
        grad_norms_per_layer[name].append(norm)
    return hook

def register_hooks(model, grad_norms_per_layer):
    hooks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            hook = param.register_hook(save_grad_norm_hook(name, grad_norms_per_layer))
            hooks.append(hook)
    return hooks

def run_pipeline(sessions, session_config, sweep_config):

    # load source session data and labels
    print("Preprocessing data...")
    channel_features_all, channel_labels_all, channel_indices = preprocess_data(sessions, session_config)

    print("Training model on source session...")
    idx_train, idx_test = train_test_split(range(len(session_config['t_starts'])), test_size=0.2, random_state=66)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.25, random_state=66)
    indices = idx_train, idx_val, idx_test
    accuracy_scores, f1_scores = [], []
    
    i = 0
    print(sessions[i])
    group = "within_session" if session_config["within_session"] else "across_session"
    data_size = len(session_config["t_starts"])
    run = wandb.init(project="diagnosis", name=f"{sessions[i]}_{group}_decoder_run",group=group, config=sweep_config, reinit=True)

    if session_config['within_session']:
        train_dataloader, val_dataloader, test_dataloader = build_single_session_graph_dataloader(channel_features_all, channel_labels_all, indices, session_config, sweep_config, GraphDataset, session_idx=i, channel_indices=channel_indices)
    else:
        train_dataloader, val_dataloader, test_dataloader = build_multi_session_graph_dataloader(channel_features_all, channel_labels_all, indices, sessions, session_config, sweep_config, GraphDataset, test_session_idx=i, channel_indices=channel_indices)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = train_dataloader.dataset[0].num_node_features
    encoder = GraphEncoder(input_size, sweep_config['latent_size'], k=sweep_config['n_layers'], tau=sweep_config['temperature']).to(device)
    model = GraphDecoder(encoder, input_dim=sweep_config['latent_size'], hidden_dim=sweep_config['decoder_layer_size'], output_dim=5).to(device)
    # if session_config['visualize']: visualize_graph(model, test_dataloader, device, test_session=sessions[i])

    supervised_criterion = nn.CrossEntropyLoss()

    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=sweep_config["encoder_learning_rate"])
    combined_optimizer = optim.Adam(model.parameters(), lr=sweep_config["decoder_learning_rate"])
    scheduler = ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.5, patience=5)
    best_decoder_acc = 0
    grad_norms_per_layer = {}
    hooks = register_hooks(model, grad_norms_per_layer)

    # initial pre-training of encoder
    for epoch in range(sweep_config["encoder_epochs"]):
        train_loss = train_graph_encoder(model.encoder, train_dataloader, encoder_optimizer, device, session_config, visualize=False)
        val_loss = validate_graph_encoder(model.encoder, val_dataloader, device, session_config)
        scheduler.step(val_loss)

        current_lr = encoder_optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{sweep_config["encoder_epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.4f}')
        wandb.log({"train_loss": train_loss, "val_loss": val_loss,  "encoder_epochs": epoch+1})
    
    plt.figure(figsize=(10, 6))
    for layer_name, grad_norms in grad_norms_per_layer.items():
        plt.plot(grad_norms, label=layer_name)
    plt.legend()
    plt.savefig('grad.png')
    
    # fine tuning encoder + decoder OR freeze encoder + train decoder
    for epoch in range(sweep_config['decoder_epochs']):
        decoder_train_metrics = train_graph_decoder(model, train_dataloader, combined_optimizer, supervised_criterion, session_config, separate=session_config["separate"], device=device)
        decoder_val_metrics = validate_graph_decoder(model, val_dataloader, supervised_criterion, device=device)

        if decoder_val_metrics['accuracy'] > best_decoder_acc:
            best_decoder_acc, patience_counter = decoder_val_metrics['accuracy'], 0
        else:
            patience_counter += 1
            if patience_counter >= sweep_config['patience']: break

        print(f'Epoch [{epoch+1}/{sweep_config["decoder_epochs"]}], Train Loss: {decoder_train_metrics["loss"]:.4f}, Val Loss: {decoder_val_metrics["loss"]:.4f}, Train Acc: {decoder_train_metrics["accuracy"]:.4f}, Val Acc: {decoder_val_metrics["accuracy"]:.4f}')
        wandb.log({"decoder_train_loss": decoder_train_metrics["loss"], "decoder_val_loss": decoder_val_metrics['loss'], 
                "decoder_train_acc": decoder_train_metrics["accuracy"], "decoder_val_acc": decoder_val_metrics['accuracy'], "decoder_epochs": epoch+1})

    file_path = os.path.join(session_config['pickle_path'], f'{session_names[i]}_{group}_graph_results.pkl')
    decoder_test_metrics = validate_graph_decoder(model, test_dataloader, supervised_criterion, device=device, pickle_path=file_path)
    print(f"target session {sessions[i]} test balanced accuracy: ", decoder_test_metrics["balanced_accuracy"],
            f"target session {sessions[i]} test f1 score: ", decoder_test_metrics["macro_f1"])

    wandb.log(decoder_test_metrics)
    accuracy_scores.append(decoder_test_metrics["balanced_accuracy"])
    f1_scores.append(decoder_test_metrics['macro_f1'])
    del train_dataloader, val_dataloader, test_dataloader
    torch.cuda.empty_cache()

    run.finish()

    return accuracy_scores
    
if __name__ == "__main__":
    wandb_config = {
        'n_layers': 2,
        'decoder_layer_size': 512,
        'encoder_epochs': 20,
        'decoder_epochs': 10,
        'patience': 10,
        'latent_size': 256,
        'spectrogram_size': 500,
        'encoder_learning_rate': 5.7e-5,
        'decoder_learning_rate': 9.7e-5,
        'batch_size': 224,
        'temperature': 0.27
    }

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    results_folder_path = "/scratch/th3129/region_decoding/results"
    session_config = {
        "separate": True,
        "visualize": True,
        'swr_flag': False,
        't_starts': np.arange(2160, 2340, 3),
        'within_session': True,
        'supervised_contrastive': False,
        'sampling_rate': 20000,
        'trial_length': 3,
        'pickle_path': results_folder_path,
        'drop_edge_rate':0.7,
        'percent_masked':0.2,
        'mask_size':1,
        'channel_idx': False
    }

    session_names = ["AD_HF01_1"]
    within_accs = run_pipeline(session_names, session_config, wandb_config)

    # session_config['within_session'] = False
    # across_accs = run_pipeline(session_names, session_config, wandb_config)
    # # within_accs, across_accs = load_results('/scratch/th3129/region_decoding/results/mlp_decoder_results.csv')
    # visualize_session_accuracy(within_accs, across_accs, session_names, labels=['within', 'across'])

    # print(within_accs, across_accs)