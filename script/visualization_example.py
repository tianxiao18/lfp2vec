import pickle
import torch
import argparse
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blind_localization.visualization_model import *

# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='SimCLR baseline')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen or ibl', required=True)
    parser.add_argument('--trial_length', type=int, default=60, help='trial_length')
    parser.add_argument('--is_within', type=bool, default=True, help='is_within')
    return parser.parse_args() 


def run_pipeline(sessions, i, is_within):
    model_name = "SimCLR_MLP" 
    within_session = "within_session_" if is_within else ""
    
    with open(f'/scratch/th3129/region_decoding/results/Neuronexus/spectrogram/SimCLR_MLP/{sessions[i]}_{within_session}sweep_config.pickle', 'rb') as f:
        sweep_config = pickle.load(f)
    print("Sweep Config: ", sweep_config)
    # sweep_config['latent_size'] = 32
    # sweep_config['decoder_layer_size'] = 128
    
    data_path = pickle_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = sweep_config['spectrogram_size'] * sweep_config['time_bins']
    
    encoder = ContrastiveEncoder(sweep_config['fc_layer_size'], input_size=input_size,
                                 output_size=sweep_config['latent_size']).to(device)
    model = ContrastiveLearningWithMLP(encoder, input_dim=sweep_config['latent_size'],
                                       hidden_dim=sweep_config['decoder_layer_size'], output_dim=5).to(device)
    ## load model
    checkpoint_path = f'/scratch/th3129/region_decoding/results/Neuronexus/spectrogram/SimCLR_MLP/{sessions[i]}_{within_session}model.pth'
    # checkpoint_path = f'/scratch/cl7201/region_decoding/results/Neuronexus/spectrogram/SimCLR_MLP/{sessions[i]}_model.pth'
    print(f"Loading model from {checkpoint_path}")
    encoder.load_state_dict(torch.load(checkpoint_path))

    # Load the saved encoder weights
    model_checkpoint_path = f'/scratch/th3129/region_decoding/results/Neuronexus/spectrogram/SimCLR_MLP/{sessions[i]}_{within_session}decoder.pth'
    # model_checkpoint_path = f'/scratch/cl7201/region_decoding/results/Neuronexus/spectrogram/SimCLR_MLP/{sessions[i]}_decoder.pth'
    model.load_state_dict(torch.load(model_checkpoint_path))

    data_type = "spectrogram"

    ######### visualization
    vis(data, data_type, model_name, encoder, model, i, sessions, data_path, output_path)
    ########


if __name__ == "__main__":
    args = arg_parser()
    data, is_within, trial_length = args.data, args.is_within, args.trial_length
    data_type = "spectrogram"
    print(f"Data: {data}, Data Type: {data_type}, Trial Length: {trial_length}, is_within: {is_within}")

    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Visual Cortex'}
    acronyms_arr = np.array(sorted(list(hc_acronyms)))
    acronyms_arr_num = np.arange(len(acronyms_arr))
    acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
    print(acr_dict)

    sessions_list = []

    if data == "Allen":
        sessions = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
        sessions_list = []
        pickle_path = "spectrogram/Allen"
        for root, dirs, files in os.walk(pickle_path):
            for file in files:
                if file.endswith('_raw.pickle') and file.split('_')[0] in sessions:
                    sessions_list.append(file.replace('_raw.pickle', ''))
    elif data == 'ibl':
        sessions_list = np.load('ibl_preprocessing/selected_eids.npy', allow_pickle=True)
        pickle_path = f'/scratch/cl7201/shared/{data}/{data_type}'
        hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
    elif data == "Neuronexus":
        sessions_list = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1","AD_HF03_2", "NN_syn_01", "NN_syn_02"]
        pickle_path = f'/scratch/cl7201/shared/{data}/{data_type}'
        hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Cortex'}

    print(f"Sessions List: {sessions_list}")
    print(f"Total number of sessions: {len(sessions_list)}")
    print(sessions_list)

    output_path = f"../results/{data}/{data_type}/SimCLR_MLP"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(len(sessions_list)):
        run_pipeline(sessions_list,i, is_within)

       