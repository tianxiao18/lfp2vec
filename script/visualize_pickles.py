import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle
import pandas as pd
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blind_localization.visualize import visualize_session_metrics
from blind_localization.data.PCAviz import PCAVisualizer

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize_embeddings(visualizer, session, train_embeddings, val_embeddings, test_embeddings,
                        train_labels, val_labels, test_labels):
    visualizer.create_pca(train_embeddings, train_labels, 2, session, "train")
    visualizer.create_pca(train_embeddings, train_labels, 3, session, "train")
    visualizer.create_pca(val_embeddings, val_labels, 2, session, "val")
    visualizer.create_pca(val_embeddings, val_labels, 3, session, "val")
    visualizer.create_pca(test_embeddings, test_labels, 2, session, "test")
    visualizer.create_pca(test_embeddings, test_labels, 3, session, "test")

    datasets = {
        "train": {"embeddings": train_embeddings, "labels": train_labels},
        "val": {"embeddings": val_embeddings, "labels": val_labels},
        "test": {"embeddings": test_embeddings, "labels": test_labels}
    }
    visualizer.create_combined_pca(datasets, 2, session, "combined")
    visualizer.create_combined_pca(datasets, 3, session, "combined")

def create_radar_plot_from_json(json_data, order, title, output_path, data_type, model):
    # Extract session IDs based on the provided order
    sessions_list = order

    # Extract values in the given order
    train_acc_list = [json_data[session]['train_acc'] for session in order]
    test_acc_list = [json_data[session]['test_acc'] for session in order]
    chance_acc_list = [json_data[session]['chance_acc'] for session in order]

    # Calculate mean and std for each metric and drop values = 0
    temp_train_acc_list = [acc for acc in train_acc_list if acc != 0]
    temp_test_acc_list = [acc for acc in test_acc_list if acc != 0]
    temp_chance_acc_list = [acc for acc in chance_acc_list if acc != 0]
    print(f"Train Acc: {temp_train_acc_list}")

    mean_train_acc = np.mean(temp_train_acc_list)
    std_train_acc = np.std(temp_train_acc_list)
    mean_test_acc = np.mean(temp_test_acc_list)
    std_test_acc = np.std(temp_test_acc_list)
    mean_chance_acc = np.mean(temp_chance_acc_list)
    std_chance_acc = np.std(temp_chance_acc_list)

    print(f"Train Acc: {mean_train_acc:.5f} ± {std_train_acc:.5f}, CA: {mean_chance_acc:.5f} ± {std_chance_acc:.5f}")
    print(f"Test Acc: {mean_test_acc:.5f} ± {std_test_acc:.5f}, CA: {mean_chance_acc:.5f} ± {std_chance_acc:.5f}")

    # Complete the loop for radar chart
    num_sessions = len(sessions_list)
    angles = np.linspace(0, 2 * np.pi, num_sessions, endpoint=False).tolist()
    angles += angles[:1]

    train_acc = train_acc_list + train_acc_list[:1]
    test_acc = test_acc_list + test_acc_list[:1]
    chance_acc = chance_acc_list + chance_acc_list[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, train_acc, linewidth=2, color='blue', label='Train Accuracy')
    ax.plot(angles, test_acc, linewidth=2, color='green', label='Test Accuracy')
    ax.plot(angles, chance_acc, linewidth=2.5, linestyle='--', color='black', label='Chance Accuracy')

    for angle, value in zip(angles[:-1], train_acc[:-1]):
        ax.annotate(f'{value:.2f}', xy=(angle, value), xytext=(10, 10),
                    textcoords='offset points', ha='center', va='center', fontsize=9, color='blue')

    for angle, value in zip(angles[:-1], test_acc[:-1]):
        ax.annotate(f'{value:.2f}', xy=(angle, value), xytext=(10, -15),
                    textcoords='offset points', ha='center', va='center', fontsize=9, color='green')

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.2, 1.2, 0.2))
    ax.set_yticklabels([f'{v:.1f}' for v in np.arange(0.2, 1.2, 0.2)], fontsize=9, color='gray')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(angles[:-1])
    for i in range(len(sessions_list)):
        sessions_list[i] = sessions_list[i][:9]
    ax.set_xticklabels(sessions_list, fontsize=10, rotation=45)

    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_path}/{title}_{data_type}_{model}_radar_accuracy.png")
    plt.close()
    print(f"Radar plot saved at {output_path}/{title}_{data_type}_{model}_radar_accuracy.png")

def arg_parser():
    parser = argparse.ArgumentParser(description='wave2vec2')
    parser.add_argument('--data', type=str, help='Dataset to use: Neuronexus, Allen, or ibl', default='Neuronexus')
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()

    if args.data == 'Neuronexus':
        walk_dir = '/scratch/th3129/region_decoding/results/Neuronexus/spectrogram_preprocessed/wave2vec2/across_session/'
        sessions = ['AD_HF01_1', 'AD_HF02_2', 'AD_HF02_4', 'AD_HF03_1', 'AD_HF03_2', 'NN_syn_01', 'NN_syn_02']
        label_json = {'0': 'Cortex', '1': 'CA1', '2': 'CA2', '3': 'CA3', '4': 'DG'}

    elif args.data == 'Allen':
        walk_dir = '/scratch/mkp6112/LFP/region_decoding/results/Allen/spectrogram/wave2vec2/across_session/'
        sessions = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
        label_json = {'0': 'CA1', '1': 'CA2', '2': 'CA3', '3': 'DG', '4': 'Cortex'}

    elif args.data == 'ibl':
        walk_dir = '/scratch/mkp6112/LFP/region_decoding/results/ibl/spectrogram/wave2vec2/across_session/'
        sessions = ['0802ced5-33a3-405e-8336-b65ebc5cb07c_probe00','0802ced5-33a3-405e-8336-b65ebc5cb07c_probe01',
                    '0a018f12-ee06-4b11-97aa-bbbff5448e9f_probe00','3638d102-e8b6-4230-8742-e548cd87a949_probe01',
                    '5dcee0eb-b34d-4652-acc3-d10afc6eae68_probe00','d2832a38-27f6-452d-91d6-af72d794136c_probe00',
                    '54238fd6-d2d0-4408-b1a9-d19d24fd29ce_probe00']
        label_json = {'0': 'CA1', '1': 'CA2', '2': 'CA3', '3': 'DG', '4': 'Cortex'}
    output_path = '/scratch/th3129/region_decoding/results/emb'

    accuracy = {}

    for root, dirs, files in os.walk(walk_dir):
        for file in files:
            current_session = file.replace('_results.pickle', '')
            
            if file.endswith('_results.pickle') and any(session in current_session for session in sessions):
                saved_data = read_pickle(walk_dir + file)
                print(saved_data.keys())

                # print the decoding accuracy 
                accuracy[current_session] = {'train_acc': saved_data['train_acc'], 'val_acc': saved_data['val_acc'], 'test_acc': saved_data['test_acc'], 'chance_acc':saved_data['test_chance_accuracy']}
                print(current_session, accuracy[current_session])

                # visualize the embedding
                visualizer = PCAVisualizer(label_json, output_path=output_path)
                visualize_embeddings(visualizer, current_session, saved_data['train_embeddings'], saved_data['val_embeddings'], saved_data['test_embeddings'],
                        saved_data['train_labels'], saved_data['val_labels'], saved_data['test_labels'])

    create_radar_plot_from_json(accuracy, sessions, args.data, '/scratch/th3129/region_decoding/results', 'accuracy', 'w2v2_repickled')

