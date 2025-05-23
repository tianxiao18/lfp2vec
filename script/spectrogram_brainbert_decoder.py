import os
import json
import torch
import torch.nn as nn
import wandb
import pickle
import numpy as np

from torch import optim
from functools import partial

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blind_localization.data.datasets import RawDataset
# from blind_localization.models.decoder import train_decoder, validate_decoder  # ContrastiveLearningWithMLP
# from blind_localization.models.contrastive import ContrastiveEncoder, InfoNCELoss
from blind_localization.models.contrastive_pipeline import build_single_session_dataloader, build_multi_session_dataloader  # , train, validation

from omegaconf import OmegaConf
from blind_localization.models.brainbert import \
    BrainBERTModel, PretrainMaskedCriterion, train, validation, \
    BrainBERTWithMLP, train_decoder, validate_decoder



def load_preprocessed_data(spectrogram_path, file_path):
    features, labels, trials = {}, {}, {}
    for session in file_path:
        data = pickle.load(open(f"{spectrogram_path}/{session}_raw.pickle", 'rb'))
        X, y, trial_idx = zip(*[(d[0], d[1], d[2]) for d in data])
        features[session] = np.array(X)
        labels[session] = np.array(y)
        trials[session] = np.array(trial_idx)
    return features, labels, trials


def run_pipeline(sessions, session_config, sweep_config, i):
    # load source session data and labels
    file_path = sessions

    print("Loading preprocessed data...")
    channel_features_all, channel_labels_all, channel_trials_all = load_preprocessed_data(spectrogram_path, file_path)

    print("Training model on source session...")
    accuracy_scores = []

    group = "within_session" if session_config["within_session"] else "across_session"
    run = wandb.init(name=f"{sessions[i]}_{group}_decoder_run", group=group, config=sweep_config, reinit=True)

    run_id = wandb.run.id
    sweep_id_json = json.load(open(sweep_id_json_path, 'r'))
    print(f"Running {sessions[i]}_{group}_decoder_run with id: {run_id}")
    if f"{sessions_list[i]}_{is_within}" not in sweep_id_json:
        sweep_id_json[f"{sessions_list[i]}_{is_within}"] = [run_id]
    else:
        sweep_id_json[f"{sessions_list[i]}_{is_within}"].append(run_id)
    with open(sweep_id_json_path, 'w') as f:
        json.dump(sweep_id_json, f)

    sweep_config = wandb.config

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
    # assert input_size == 2000, input_size


    # print(type(train_dataloader.dataset))  # <class 'blind_localization.data.datasets.RawDataset'>
    # print(type(train_dataloader.dataset[0]), len(train_dataloader.dataset[0]))  # <class 'tuple'> 2
    # print(type(train_dataloader.dataset[0][0]), len(train_dataloader.dataset[0][0]))  # <class 'tuple'> 2
    # print(type(train_dataloader.dataset[0][1]), train_dataloader.dataset[0][1])  # <class 'torch.Tensor'>
    # print(type(train_dataloader.dataset[0][0][0]), train_dataloader.dataset[0][0][0].shape)  # <class 'torch.Tensor'> torch.Size([2000])


    model_cfg = OmegaConf.create({ "hidden_dim": sweep_config["encoder_nhead"] * sweep_config["encoder_hid_dim_factor"],
                                   "layer_dim_feedforward": sweep_config["encoder_ffd_dim"],
                                   "layer_activation": "gelu",
                                   "nhead": sweep_config["encoder_nhead"],
                                   "encoder_num_layers": sweep_config["encoder_num_layers"],
                                   "input_dim": sweep_config["spectrogram_size"] })
    encoder = BrainBERTModel()
    encoder.build_model(model_cfg)
    encoder.to("cuda")
    input_dim = input_size // sweep_config["spectrogram_size"] * sweep_config["encoder_nhead"] * sweep_config["encoder_hid_dim_factor"]
    model = BrainBERTWithMLP(encoder, input_dim=input_dim,
                             hidden_dim=sweep_config["decoder_layer_size"], output_dim=5).to(device)


    # initial pre-training of encoder
    encoder_criterion = PretrainMaskedCriterion(alpha=2)
    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=sweep_config["encoder_learning_rate"])
    for epoch in range(sweep_config["encoder_epochs"]):
        train_loss = train(model.encoder, train_dataloader, encoder_optimizer, encoder_criterion, device)
        val_loss = validation(model.encoder, val_dataloader, encoder_criterion, device)

        print(
            f'Epoch [{epoch + 1}/{sweep_config["encoder_epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "encoder_epochs": epoch + 1})


    # fine tuning encoder + decoder OR freeze encoder + train decoder
    best_val_acc = 0
    best_model_state = None
    combined_criterion = nn.CrossEntropyLoss()
    combined_optimizer = optim.Adam(model.parameters(), lr=sweep_config["decoder_learning_rate"])
    for epoch in range(sweep_config['decoder_epochs']):
        decoder_train_loss, decoder_train_acc = train_decoder(model, train_dataloader, combined_optimizer,
                                                              encoder_criterion, combined_criterion,
                                                              mode=session_config['separate'], device=device)
        decoder_val_loss, decoder_val_acc, decoder_val_f1_score = validate_decoder(model, val_dataloader, combined_criterion, device=device)

        if decoder_val_acc > best_val_acc:
            best_val_acc, patience_counter = decoder_val_acc, 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= sweep_config["patience"]: break

        print(
            f'Epoch [{epoch + 1}/{sweep_config["decoder_epochs"]}], Train Loss: {decoder_train_loss:.4f}, '
            f'Val Loss: {decoder_val_loss:.4f}, Train Acc: {decoder_train_acc:.4f}, Val Acc: {decoder_val_acc:.4f}, '
            f'Val F1 Score: {decoder_val_f1_score:.4f}')
        wandb.log({"decoder_train_loss": decoder_train_loss, "decoder_val_loss": decoder_val_loss,
                   "decoder_train_acc": decoder_train_acc, "decoder_val_acc": decoder_val_acc,
                   "decoder_val_f1_score": decoder_val_f1_score, "decoder_epochs": epoch + 1})

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("No improvement during training, using the last model for testing.")
    _, decoder_test_acc, decoder_test_f1_score = validate_decoder(model, test_dataloader, combined_criterion, device=device)
    print(f"target session {sessions[i]} test accuracy: {decoder_test_acc} test f1 score: {decoder_test_f1_score}")
    wandb.log({"test_acc": decoder_test_acc, "test_f1_score": decoder_test_f1_score})
    accuracy_scores.append(decoder_test_acc)

    run.finish()

    return accuracy_scores


def run_pipeline_wrapper(i, is_within):
    session_config = {
        "separate": False,
        "visualize": True,
        'swr_flag': False,
        'within_session': is_within,
        'supervised_contrastive': False,
        'sampling_rate': 20000,
        'trial_length': 0.5
    }

    test_acc = run_pipeline(sessions_list, session_config, wandb.config, i)
    return test_acc


if __name__ == "__main__":
    is_within = True
    sweep_config = {
        "method": "bayes",
        "metric": { "name": "test_acc", "goal": "maximize" }
    }
    parameters_dict = {

        'batch_size': {'distribution': 'q_log_uniform_values', 'q': 16, 'min': 16, 'max': 192},

        "encoder_ffd_dim": {'distribution': 'q_log_uniform_values', 'q': 32, 'min': 32, 'max': 2048},
        "encoder_hid_dim_factor": {'distribution': 'q_log_uniform_values', 'q': 8, 'min': 8, 'max': 48},
        "encoder_nhead": {'values': [ 4, 6, 8, 10, 12 ]},
        "encoder_num_layers": {'values': [ 3, 4, 5, 6, 7 ]},
        'encoder_learning_rate': {'distribution': 'uniform', 'min': 3e-5, 'max': 2e-4},

        'decoder_layer_size': {'distribution': 'q_log_uniform_values', 'q': 32, 'min': 32, 'max': 256},
        'decoder_learning_rate': {'distribution': 'uniform', 'min': 1e-5, 'max': 1e-4},

        'encoder_epochs': {'values': [ 100 ]},
        'decoder_epochs': {'values': [ 100 ]},
        'patience': {'values': [ 20 ]},
        'spectrogram_size': {'values': [ 500 ]},
    }
    sweep_config['parameters'] = parameters_dict

    # Create a sweep id that stores sweep ids
    sweep_id_json_path = 'sweep_id.json'
    if not os.path.exists(sweep_id_json_path):
        with open(sweep_id_json_path, 'w') as f:
            json.dump({}, f)
    sweep_id_json = json.load(open(sweep_id_json_path, 'r'))

    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Visual Cortex'}
    acronyms_arr = np.array(sorted(list(hc_acronyms)))
    acronyms_arr_num = np.arange(len(acronyms_arr))
    acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
    print(acr_dict)

    manifest_path = 'data'
    if not os.path.exists(manifest_path):
        os.makedirs(manifest_path)

    sessions_list = []

    print(f"Current working directory: {os.getcwd()}")

    spectrogram_path = "spectrogram/Allen"
    output_path = "../results/Allen/spectrogram/MLP"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for root, dirs, files in os.walk(spectrogram_path):
        for file in files:
            if file.endswith('_raw.pickle'):
                sessions_list.append(file.replace('_raw.pickle', ''))

    print(f"Total number of sessions: {len(sessions_list)}")
    print(sessions_list)

    for i in range(len(sessions_list)):
        run_pipeline_with_args = partial(run_pipeline_wrapper, i, is_within)

        if f"{sessions_list[i]}_{is_within}" not in sweep_id_json:
            sweep_id = wandb.sweep(sweep_config, project="spectrogram_brainbert_inter_mlp_2")
        else:
            sweep_id = wandb.sweep(sweep_config, project="spectrogram_brainbert_inter_mlp_2",
                                   prior_runs=sweep_id_json[f"{sessions_list[i]}_{is_within}"])

        # print(sweep_id)
        wandb.agent(sweep_id, run_pipeline_with_args, count=2)
