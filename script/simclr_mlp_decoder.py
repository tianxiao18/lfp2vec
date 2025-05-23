import os
import time
import wandb
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from script.utils import save_if_better
from functools import partial
from blind_localization.data.datasets import RawDataset
from blind_localization.models.contrastive_pipeline import *
from blind_localization.models.contrastive import *
from blind_localization.models.decoder import *


#######################################################################################################################
# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='SimCLR baseline')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen or ibl', default="Monkey")
    parser.add_argument('--trial_length', type=int, default=60, help='trial_length')
    parser.add_argument('--data_type', type=str, help='Data type to use', default='spectrogram')
    parser.add_argument('--session_list', type=str, help='List of sessions to use')
    parser.add_argument('--within_session', type=lambda x: x.lower() == 'true', default=False,
                        help='Within or across session')
    return parser.parse_args()


args = arg_parser()
data, trial_length, data_type, args_session_list, within_session = (
    args.data, args.trial_length, args.data_type, args.session_list, args.within_session)
print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}, Within Session: {within_session}")

#############

if data == "Allen":
    # sessions = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
    # sessions = ['719161530']
    all_sessions = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
    if args_session_list is None or args_session_list == '':
        sessions = all_sessions
    else:
        sessions = args_session_list.split(',')
        all_sessions = args_session_list.split(',')

    sessions_list = all_sessions

    pickle_path = "spectrogram/Allen"
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Visual Cortex'}
elif data == 'ibl':
    all_sessions = ['0802ced5-33a3-405e-8336-b65ebc5cb07c_probe00',
                    '0802ced5-33a3-405e-8336-b65ebc5cb07c_probe01',
                    '0a018f12-ee06-4b11-97aa-bbbff5448e9f_probe00',
                    '3638d102-e8b6-4230-8742-e548cd87a949_probe01',
                    '5dcee0eb-b34d-4652-acc3-d10afc6eae68_probe00',
                    'd2832a38-27f6-452d-91d6-af72d794136c_probe00',
                    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce_probe00']

    if args_session_list is None or args_session_list == '':
        sessions = all_sessions
    else:
        sessions = args_session_list.split(',')

    sessions_list = sessions
    pickle_path = f'/vast/th3129/data/ibl_new/spectrogram_preprocessed'
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
elif data == "Neuronexus":
    all_sessions = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]

    if args_session_list is None or args_session_list == '':
        sessions = all_sessions
    else:
        sessions = args_session_list.split(',')

    sessions_list = sessions

    pickle_path = f'/scratch/th3129/region_decoding/data/Neuronexus/lfp'
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Visual Cortex'}
elif data == "Monkey":
    all_sessions = ['221007', '221216', '221104']

    if args_session_list is None or args_session_list == '':
        sessions = all_sessions
    else:
        sessions = args_session_list.split(',')

    sessions_list = sessions

    pickle_path = "spectrogram/Allen"
    hc_acronyms = ['Basal_Ganglia', 'Suppl_Motor_Area', 'Primary_Motor_Cortex']

acronyms_arr = np.array(list(hc_acronyms))
acronyms_arr_num = np.arange(len(acronyms_arr))
acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
print(acr_dict)

print(f"Sessions List: {sessions_list}")
print(f"Total number of sessions: {len(sessions_list)}")
print(sessions_list)

output_path = f"../results/{data}/{data_type}/SimCLR_MLP"
if not os.path.exists(output_path):
    os.makedirs(output_path)


def load_preprocessed_data(pickle_path, file_path, data_type='raw'):
    features, labels, trials, chans = {}, {}, {}, {}
    for session in file_path:
        if args.data == "Monkey":
            data = pickle.load(open(f"{pickle_path}/{session}.pickle", 'rb'))
        elif data_type == 'raw' and args.data != "Neuronexus":
            data = pickle.load(open(f"{pickle_path}/{session}_raw.pickle", 'rb'))
        elif data_type == 'lfp' or args.data == "Neuronexus":
            data = pickle.load(open(f"{pickle_path}/{session}_lfp.pickle", 'rb'))
        X, y, trial_idx, chan_id = zip(*[(d[0], d[1], d[2], d[3]) for d in data])
        if args.data == "Monkey":
            y = [label.replace("imec", "") for label in y]
        features[session] = np.array(X)
        labels[session] = np.array(y, dtype=int)
        trials[session] = np.array(trial_idx)
        chans[session] = np.array(chan_id)

        non_zero_indices = [i for i, x in enumerate(features[session]) if not np.all(x == 0)]

        features[session] = features[session][non_zero_indices]
        labels[session] = labels[session][non_zero_indices]
        trials[session] = trials[session][non_zero_indices]
        chans[session] = chans[session][non_zero_indices]

        # Sanity check
        assert len(features[session]) == len(labels[session]) == len(trials[session]) == len(chans[session]), \
            f"Inconsistent data sizes for session {session}"

    return features, labels, trials, chans


def run_pipeline(sessions, session_config, pipeline_sweep_config, i):
    # load source session data and labels
    file_path = sessions

    print("Loading preprocessed data...")
    channel_features_all, channel_labels_all, channel_trials_all, channel_chans_all = load_preprocessed_data(
        pickle_path, file_path)

    print("Training model on source session...")

    group = "within_session" if session_config["within_session"] else "across_session"
    run = wandb.init(name=f"{sessions[i]}_{group}_decoder_run", group=group, config=pipeline_sweep_config, reinit=True)

    run_id = wandb.run.id
    sweep_id_json = json.load(open(sweep_id_json_path, 'r'))
    print(f"Running {sessions[i]}_{group}_decoder_run with id: {run_id}")
    if f"{sessions[i]}_{is_within}" not in sweep_id_json:
        sweep_id_json[f"{sessions[i]}_{is_within}"] = [run_id]
    else:
        sweep_id_json[f"{sessions[i]}_{is_within}"].append(run_id)
    with open(sweep_id_json_path, 'w') as f:
        json.dump(sweep_id_json, f)

    pipeline_sweep_config = wandb.config

    if session_config['within_session']:
        train_dataloader, val_dataloader, test_dataloader = build_single_session_dataloader(channel_features_all,
                                                                                            channel_labels_all,
                                                                                            channel_trials_all,
                                                                                            pipeline_sweep_config,
                                                                                            RawDataset, sessions,
                                                                                            session_idx=i)
    else:
        # train_dataloader, val_dataloader, test_dataloader = build_multi_session_dataloader(channel_features_all,
        #                                                                                    channel_labels_all,
        #                                                                                    channel_trials_all,
        #                                                                                    sessions,
        #                                                                                    pipeline_sweep_config,
        #                                                                                    RawDataset,
        #                                                                                    test_session_idx=i)
        train_dataloader, val_dataloader, test_dataloader, train_metadata, val_metadata, test_metadata = build_multi_session_dataloader_new(
            channel_features_all,
            channel_labels_all,
            channel_trials_all,
            channel_chans_all,
            sessions,
            pipeline_sweep_config,
            RawDataset,
            test_session_idx=i)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = train_dataloader.dataset[0][0][0].size()[0]
    output_dim = 5 if args.data != "Monkey" else 3
    encoder = ContrastiveEncoder(pipeline_sweep_config['fc_layer_size'], input_size=input_size,
                                 output_size=pipeline_sweep_config['latent_size']).to(device)
    model = ContrastiveLearningWithMLP(encoder, input_dim=pipeline_sweep_config['latent_size'],
                                       hidden_dim=pipeline_sweep_config['decoder_layer_size'], output_dim=output_dim).to(device)

    contrastive_criterion = InfoNCELoss(temperature=pipeline_sweep_config['temperature'], device=device)
    supervised_criterion = nn.CrossEntropyLoss()

    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=pipeline_sweep_config["encoder_learning_rate"])
    combined_optimizer = optim.Adam(model.parameters(), lr=pipeline_sweep_config["decoder_learning_rate"])
    best_decoder_acc = 0

    # initial pre-training of encoder
    for epoch in range(pipeline_sweep_config["encoder_epochs"]):
        start_time = time.time()
        train_loss = train(model.encoder, train_dataloader, encoder_optimizer, contrastive_criterion, device)
        val_loss = validation(model.encoder, val_dataloader, contrastive_criterion, device)

        print(f'Epoch [{epoch + 1}/{pipeline_sweep_config["encoder_epochs"]}], Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Time: {time.time() - start_time:.4f}')
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "encoder_epochs": epoch + 1})

    # fine tuning encoder + decoder OR freeze encoder + train decoder
    for epoch in range(pipeline_sweep_config['decoder_epochs']):
        start_time = time.time()
        decoder_train_metrics = train_decoder(model, train_dataloader, combined_optimizer,
                                              contrastive_criterion, supervised_criterion,
                                              mode=session_config['separate'], device=device)
        decoder_metrics = validate_decoder(model, val_dataloader,
                                           supervised_criterion,
                                           device=device)

        decoder_train_loss = float(decoder_train_metrics['loss'])
        decoder_train_acc = float(decoder_train_metrics['accuracy'])

        decoder_val_acc = float(decoder_metrics['balanced_accuracy'])
        decoder_val_loss = float(decoder_metrics['loss'])
        decoder_val_f1_score = float(decoder_metrics['macro_f1'])

        if decoder_val_acc > best_decoder_acc:
            best_val_acc, patience_counter = decoder_val_acc, 0
        else:
            patience_counter += 1
            if patience_counter >= pipeline_sweep_config['patience']: break

        print(f"Epoch [{epoch + 1}/{pipeline_sweep_config['decoder_epochs']}], Train Loss: {decoder_train_loss:.4f}, "
              f"Val Loss: {decoder_val_loss:.4f}, Train Acc: {decoder_train_acc:.4f}, Val Acc: {decoder_val_acc:.4f}, "
              f"Val F1 Score: {decoder_val_f1_score:.4f}, Time: {time.time() - start_time:.4f}")
        wandb.log({"decoder_train_loss": decoder_train_loss, "decoder_val_loss": decoder_val_loss,
                   "decoder_train_acc": decoder_train_acc, "decoder_val_acc": decoder_val_acc,
                   "decoder_val_f1_score": decoder_val_f1_score, "decoder_epochs": epoch + 1})

    decoder_train_metrics = validate_decoder(model, train_dataloader, supervised_criterion, device=device)
    decoder_train_acc = float(decoder_train_metrics['balanced_accuracy'])
    decoder_train_f1_score = float(decoder_train_metrics['macro_f1'])
    train_real_labels, train_pred_labels = (
        decoder_train_metrics['real_labels'], decoder_train_metrics['predicted_labels'])

    print(f"target session {sessions[i]:} train accuracy: {decoder_train_acc:.5f} "
          f"train f1 score: {decoder_train_f1_score:.5f}")
    wandb.log({"train_acc": decoder_train_acc, "train_f1_score": decoder_train_f1_score})

    decoder_val_metrics = validate_decoder(model, val_dataloader, supervised_criterion, device=device)
    decoder_test_metrics = validate_decoder(model, test_dataloader, supervised_criterion, device=device)
    decoder_test_acc = float(decoder_test_metrics['balanced_accuracy'])
    decoder_test_f1_score = float(decoder_test_metrics['macro_f1'])
    test_real_labels, test_pred_labels = (
        decoder_test_metrics['real_labels'], decoder_test_metrics['predicted_labels'])
    print(f"target session {sessions[i]} test accuracy: {decoder_test_acc:.5f} "
          f"test f1 score: {decoder_test_f1_score:.5f}")
    wandb.log({"test_acc": decoder_test_acc, "test_f1_score": decoder_test_f1_score})

    val_acc = 0
    if session_config['within_session']:
        sess_type = 'within'
    else:
        sess_type = 'across'
    if os.path.exists(f"{output_path}/{sessions[i]}_{sess_type}_session.pickle"):
        # Load the existing pickle file
        with open(f"{output_path}/{sessions[i]}_{sess_type}_session.pickle", 'rb') as f:
            saved_data = pickle.load(f)
            val_acc = saved_data['decoder_val_acc']

    # Save encoder, model and sweep_config
    if best_val_acc > val_acc:
        torch.save(encoder.state_dict(), f"{output_path}/{sessions[i]}_{sess_type}_session_model.pth")
        torch.save(model.state_dict(), f"{output_path}/{sessions[i]}_{sess_type}_session_decoder.pth")

        params = ['fc_layer_size', 'decoder_layer_size', 'encoder_epochs', 'decoder_epochs', 'patience', 'latent_size',
                  'spectrogram_size', 'encoder_learning_rate', 'decoder_learning_rate', 'batch_size', 'temperature',
                  'time_bins', 'library', 'dataset']
        sweep_dict = [pipeline_sweep_config[key] for key in params if key in pipeline_sweep_config]

        print(len(train_metadata), train_metadata[0])
        print(np.array(decoder_train_metrics['logits']).shape)
        print(np.array(decoder_train_metrics['embeddings']).shape)

        file_path = os.path.join(output_path, f'{sessions[i]}_results.pickle')
        file_obj = {
            'train_logits': decoder_train_metrics['logits'],
            'train_labels': decoder_train_metrics['real_labels'],
            'train_acc': decoder_train_metrics['accuracy'],
            'train_embeddings': decoder_train_metrics['embeddings'],
            'train_trial_chans': train_metadata,
            'val_logits': decoder_val_metrics['logits'],
            'val_labels': decoder_val_metrics['real_labels'],
            'val_acc': decoder_val_metrics['accuracy'],
            'val_embeddings': decoder_val_metrics['embeddings'],
            'val_trial_chans': val_metadata,
            'test_logits': decoder_test_metrics['logits'],
            'test_labels': decoder_test_metrics['real_labels'],
            'test_acc': decoder_test_metrics['accuracy'],
            'test_embeddings': decoder_test_metrics['embeddings'],
            'test_trial_chans': test_metadata,
            'simclr_config': sweep_dict
        }
        with open(file_path, 'wb') as f:
            pickle.dump(file_obj, f)
            print(f"Session {sessions[i]} results saved to {file_path}")
            print(
                f"Train acc: {file_obj['train_acc']}, Val acc: {file_obj['val_acc']}, Test acc: {file_obj['test_acc']}")

    run.finish()

    return train_real_labels, train_pred_labels, test_real_labels, test_pred_labels, val_acc


def run_pipeline_wrapper(i, is_within, output_path):
    session_config = {
        "separate": "separate",
        "visualize": True,
        'swr_flag': False,
        'within_session': is_within,
        'supervised_contrastive': False
    }
    output_path = f"{output_path}/{session_config['separate']}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_real_labels, train_pred_labels, test_real_labels, test_pred_labels, decoder_val_acc = (
        run_pipeline(all_sessions, session_config, wandb.config, i))

    val_acc = round(decoder_val_acc, 5)

    if session_config['within_session']:
        title = f"{all_sessions[i]}_within_session"
    else:
        title = f"{all_sessions[i]}_across_session"

    output_path = f"{output_path}/{title}.pickle"
    save_if_better(output_path, val_acc, train_real_labels, train_pred_labels, test_real_labels, test_pred_labels)

    return None


if __name__ == "__main__":
    is_within = within_session

    sweep_config = {'method': 'random'}
    parameters_dict = {
        'fc_layer_size': {'values': [128, 256, 512]},
        'decoder_layer_size': {'values': [128, 256, 512]},
        'encoder_epochs': {'values': [250]},
        'decoder_epochs': {'values': [100, 120, 150]},
        'patience': {'values': [5, 10, 15]},
        'latent_size': {'values': [4, 8, 16, 32, 64, 128, 256, 512]},
        'spectrogram_size': {'values': [500]},
        'encoder_learning_rate': {'distribution': 'uniform', 'min': 1e-5, 'max': 1e-4},
        'decoder_learning_rate': {'distribution': 'uniform', 'min': 1e-5, 'max': 1e-4},
        'batch_size': {'distribution': 'q_log_uniform_values', 'q': 8, 'min': 32, 'max': 256},
        'temperature': {'distribution': 'uniform', 'min': 0.1, 'max': 0.9},
        'time_bins': {'values': [16]},
        'library': {'values': ['librosa']},
        'dataset': {'values': [data]}
    }
    sweep_config['parameters'] = parameters_dict

    if is_within:
        sess = "within"
    else:
        sess = "across"

    # Create a sweep id that stores sweep ids
    sweep_id_json_path = 'sweep_id.json'
    if not os.path.exists(sweep_id_json_path):
        with open(sweep_id_json_path, 'w') as f:
            json.dump({}, f)
    sweep_id_json = json.load(open(sweep_id_json_path, 'r'))

    for i in range(len(sessions_list)):
        # Index of sessions_list[i] in all_sessions
        sess_idx = all_sessions.index(sessions_list[i])

        run_pipeline_with_args = partial(run_pipeline_wrapper, sess_idx, is_within, output_path)

        if f"{all_sessions[sess_idx]}_{is_within}" not in sweep_id_json:
            sweep_id = wandb.sweep(sweep_config, project=f"HPC_SimCLR_{sess}_session_{data}_{data_type}")
        else:
            sweep_id = wandb.sweep(sweep_config, project=f"HPC_SimCLR_{sess}_session_{data}_{data_type}"
                                   , prior_runs=sweep_id_json[f"{all_sessions[sess_idx]}_{is_within}"])

        wandb.agent(sweep_id, run_pipeline_with_args, count=10)
