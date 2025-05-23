import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blind_localization.visualize import visualize_session_metrics

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

    # Annotating the values on the plot
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

    # plt.title(f'{title} - Accuracy (Radar Plot) ({data_type})', fontsize=14, y=1.1)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{title}_{data_type}_{model}_radar_accuracy.png")
    plt.close()
    print(f"Radar plot saved at {output_path}/{title}_{data_type}_{model}_radar_accuracy.png")


def create_bar_plot_from_csv(file_path, sessions, output_path):
    df = pd.read_csv(file_path)
    labels = df['Models'].tolist()
    df = df.drop(df.columns[0], axis=1)
    accuracy_metrics = df.to_numpy()
    colors = ['#d43000','#ed9051','#7ec282','#74abcc','#3072b0']
    visualize_session_metrics(accuracy_metrics, sessions, labels, title=output_path, colors=colors)

def create_bar_plot_from_wandb_logs(file_path, sessions, output_path):
    df = pd.read_csv(file_path)
    df = df[df['Name'].notna() & df['test/accuracy'].notna()]
    df['session'] = df['Name'].apply(lambda x: x.split('-')[0])
    df['pretrained'] = df['Name'].apply(lambda x: x.split('-')[1])
    df['ssl'] = df['Name'].apply(lambda x: x.split('-')[2])
    settings = [('pretrained', 'ssl'),('pretrained', 'nossl'),('rand_init', 'ssl'),('rand_init', 'nossl')]
    # settings = [('pretrained', 'nossl'),('rand_init', 'nossl')]
    accuracy_metrics = np.zeros((4, len(sessions)))

    for i, (pt, ssl) in enumerate(settings):
        for j, session in enumerate(sessions):
            row = df[(df['session'] == session) &(df['pretrained'] == pt) &(df['ssl'] == ssl)]
            if not row.empty:
                accuracy_metrics[i][j] = row['test/accuracy'].values[0]

    print(accuracy_metrics)
    colors = ['#d43000','#ed9051','#7ec282','#74abcc','#3072b0']
    labels = ['_'.join(s) for s in settings]
    visualize_session_metrics(accuracy_metrics, sessions, labels, title=output_path, colors=colors)


if __name__ == "__main__":
  data = {}
#   walk_dir = '/scratch/th3129/region_decoding/results/Neuronexus/spectrogram_preprocessed/wave2vec2/across_session/'
#   walk_dir = '/scratch/mkp6112/LFP/region_decoding/results/Neuronexus/spectrogram/wave2vec2/across_session/'
  walk_dir = '/scratch/th3129/region_decoding/results/ibl/spectrogram_preprocessed/wave2vec2/across_session/'

  sessions = [
            '0802ced5-33a3-405e-8336-b65ebc5cb07c_probe00',
            '0802ced5-33a3-405e-8336-b65ebc5cb07c_probe01',
            '0a018f12-ee06-4b11-97aa-bbbff5448e9f_probe00',
            '3638d102-e8b6-4230-8742-e548cd87a949_probe01',
            'd2832a38-27f6-452d-91d6-af72d794136c_probe00',
            '5dcee0eb-b34d-4652-acc3-d10afc6eae68_probe00',
            '54238fd6-d2d0-4408-b1a9-d19d24fd29ce_probe00'
        ]
#   chance_accs = [0.52705, 0.41359, 0.47569, 0.40413, 0.46923, 0.34775, 0.34775]
#   chance_accs = dict(zip(sessions, chance_accs))

  for root, dirs, files in os.walk(walk_dir):
      for file in files:
          current_session = file.replace('_results.pickle', '')
          if file.endswith('_results.pickle') and any(session in current_session for session in sessions):
              with open(walk_dir + file, 'rb') as f:
                  saved_data = pickle.load(f)
                  if 'train_acc' in saved_data.keys():
                    saved_train_acc = saved_data['train_acc']
                    saved_val_acc = saved_data['val_acc']
                    saved_test_acc = saved_data['test_acc']
                    saved_chance_acc = saved_data['test_chance_accuracy']
                    data[current_session] = {'train_acc': saved_train_acc, 'val_acc': saved_val_acc, 'test_acc': saved_test_acc, 'chance_acc':saved_chance_acc}
                    print(current_session, data[current_session])

  for session in sessions:
      if session not in data:
        data[session] = {'train_acc': 0, 'val_acc': 0, 'test_acc': 0, 'chance_acc':0}

  create_radar_plot_from_json(data, sessions, 'ibl', '/scratch/th3129/region_decoding/results', 'accuracy', 'w2v2_ibl_lfp_final')
  # create_radar_plot_from_json(data, sessions, 'Neuronexus', '/scratch/th3129/region_decoding/results', 'accuracy', 'w2v2_nn_rand_init')

  sessions = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]

  input_path = '/scratch/th3129/region_decoding/results/neuronexus_within.csv'
  output_path = '/scratch/th3129/region_decoding/results/neuronexus_summary_within'
  create_bar_plot_from_csv(input_path, sessions, output_path)

  input_path = '/scratch/th3129/region_decoding/results/neuronexus_across.csv'
  output_path = '/scratch/th3129/region_decoding/results/neuronexus_summary_across_lfp'
  create_bar_plot_from_csv(input_path, sessions, output_path)

  input_path = '/scratch/th3129/region_decoding/results/ablation_exp.csv'
  output_path = '/scratch/th3129/region_decoding/results/ablation_exp'
  create_bar_plot_from_wandb_logs(input_path, sessions, output_path)
