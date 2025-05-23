import argparse
import os
import pickle
import random

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.signal import resample
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence, Value
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer, pipeline
from matplotlib.patches import Wedge
from scipy.special import softmax

# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='wave2vec2')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen or ibl', default='Neuronexus')
    return parser.parse_args()

def run_wave2vec2(sessions, sess, result_path):
    session_list = sessions
    session = sess
    print(f"Session list: {session_list}")
    
    file_path = f'{result_path}/{session}_results.pickle'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    train_labels = data['train_labels']
    train_logits = data['train_logits']
    print(f"train_labels: {train_labels}, train_logits: {train_logits}")
    val_labels = data['val_labels']
    val_logits = data['val_logits']
    print(f"val_labels: {val_labels}, val_logits: {val_logits}")
    test_labels = data['test_labels']
    test_logits = data['test_logits']
    print(f"test_labels: {test_labels}, test_logits: {test_logits}")
    #########################################################
    # visualize in pie chart
    # vis_pie_chart_result(train_logits, train_labels, 'train', f'{session}_train_pie_chart.png')
    # vis_pie_chart_result(val_logits, val_labels, 'val', f'{session}_val_pie_chart.png')
    vis_pie_chart_result(test_logits, test_labels, 'test', f'{session}_test_pie_chart.png')



def vis_pie_chart_result(logits, labels, mode, figure_name):
    print("drawing pie chart...")
    if mode == 'train':
        num_trials = int(60*0.8*0.75) + 1
        num_channels = int(len(logits)/num_trials) + 1
    elif mode == 'val':
        num_trials = int(60*0.8*0.25) + 1
        num_channels = int(len(logits)/num_trials) + 1
    elif mode == 'test':
        num_trials = int(60*0.2) + 1
        num_channels = int(len(logits)/num_trials) + 1

    print(f"num_trials: {num_trials}, num_channels: {num_channels}")
    
    region_colors = {
        "CA1": "red",
        "CA2": "blue", 
        "CA3": "green",
        "DG": "purple",
        "Visual Cortex": "orange"
    }
    regions = list(region_colors.keys())
    
    # Adjust figsize based on num_trials and num_channels
    fig, (ax, ax_label) = plt.subplots(1, 2, figsize=(int(num_trials * 0.4), int(num_channels * 0.16)))  # Create two subplots
    pie_size = 0.4  # Increase pie size for better visibility
    row = 0
    col = 0
    for idx, (logit, label) in tqdm(enumerate(zip(logits, labels)), total=len(logits), desc="Processing"):
        probabilities = softmax(logit)
        angles = probabilities * 360
        start_angle = 0
        
        for region_idx, (region, angle) in enumerate(zip(regions, angles)):
            if angle > 0:
                wedge = Wedge((col, row), pie_size, start_angle, 
                            start_angle + angle, facecolor=region_colors[region],
                            alpha=0.8, linewidth=0)
                ax.add_patch(wedge)
                start_angle += angle
        
        # Add a circle for the label
        region_name = regions[label]
        circle = plt.Circle((col, row), pie_size, color=region_colors[region_name], alpha=0.8)
        ax_label.add_patch(circle)
        
        col += 1
        if col >= num_trials:  # Ensure col resets correctly
            col = 0
            row += 1
    
    ax.set_xlim(-1, num_trials)  # Set limits to match trial indices
    ax.set_ylim(-1, num_channels)  # Set limits to match channel indices
    ax.set_xlabel("Trial Index")
    ax.set_ylabel("Channel Index")
    ax.set_title(f"{mode} predicted")

    ax_label.set_xlim(-1, num_trials)  # Set limits to match trial indices
    ax_label.set_ylim(-1, num_channels)  # Set limits to match channel indices
    ax_label.set_xlabel("Trial Index")
    ax_label.set_ylabel("Channel Index")
    # subtitle
    ax_label.set_title(f"{mode} ground truth")

    plt.tight_layout()  # Adjust layout to be tight
    plt.savefig(figure_name, dpi=100, bbox_inches='tight')  # Save with tight bounding box

if __name__ == "__main__":
    args = arg_parser()
    data = args.data
    print(f"Data: {data}")

    if data == "Allen":
        sessions_list = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
        result_path = "/scratch/mkp6112/LFP/region_decoding/results/Allen/spectrogram/wave2vec2/across_session"
    elif data == 'ibl':
        sessions_list = [
            '0802ced5-33a3-405e-8336-b65ebc5cb07c_probe00',
            '0802ced5-33a3-405e-8336-b65ebc5cb07c_probe01',
            '0a018f12-ee06-4b11-97aa-bbbff5448e9f_probe00',
            '15763234-d21e-491f-a01b-1238eb96d389_probe00',
            '3638d102-e8b6-4230-8742-e548cd87a949_probe01',
            '5dcee0eb-b34d-4652-acc3-d10afc6eae68_probe00',
            'd2832a38-27f6-452d-91d6-af72d794136c_probe00'
        ]
        result_path = '/scratch/th3129/region_decoding/results/ibl/spectrogram_preprocessed/wave2vec2/across_session'
        hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
    elif data == "Neuronexus":
        sessions_list = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]
        result_path ='/scratch/mkp6112/LFP/region_decoding/results/Neuronexus/spectrogram/wave2vec2/across_session'
        hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Cortex'}

    for i in range(2, len(sessions_list)):
        run_wave2vec2(sessions_list, sessions_list[i], result_path)

        