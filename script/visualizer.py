import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter


def plot_confusion_matrices(train_real_labels_all, train_pred_labels_all, test_real_labels_all, test_pred_labels_all,
                            sessions_list, title):
    # Create confusion matrices for each session and plot in a grid
    num_sessions = len(sessions_list)
    cols = 4  # Number of plots per row
    rows = (num_sessions + cols - 1) // cols  # Calculate number of rows needed

    # Plot train confusion matrices in a separate figure
    fig_train, axes_train = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    # fig_train.suptitle(f'{title} - Train Confusion Matrices by Session', fontsize=16)

    for i, (train_real, train_pred) in enumerate(zip(train_real_labels_all, train_pred_labels_all)):
        cm_train = confusion_matrix(train_real, train_pred)
        ax = axes_train[i // cols, i % cols] if rows > 1 else axes_train[i % cols]

        ConfusionMatrixDisplay(cm_train).plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
        ax.set_title(f'Session: {sessions_list[i][:9]}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    for j in range(i + 1, rows * cols):
        fig_train.delaxes(axes_train[j // cols, j % cols] if rows > 1 else axes_train[j % cols])

    plt.tight_layout()
    plt.savefig(f"{output_path}/{title}_{data_type}_{model}_train_confusion_matrices_by_session.png")
    plt.close(fig_train)

    # Plot test confusion matrices in a separate figure
    fig_test, axes_test = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    # fig_test.suptitle(f'{title} - Test Confusion Matrices by Session', fontsize=16)

    for i, (test_real, test_pred) in enumerate(zip(test_real_labels_all, test_pred_labels_all)):
        cm_test = confusion_matrix(test_real, test_pred)
        ax = axes_test[i // cols, i % cols] if rows > 1 else axes_test[i % cols]

        ConfusionMatrixDisplay(cm_test).plot(ax=ax, colorbar=False, cmap='Greens', values_format='d')
        ax.set_title(f'Session: {sessions_list[i][:9]}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    for j in range(i + 1, rows * cols):
        fig_test.delaxes(axes_test[j // cols, j % cols] if rows > 1 else axes_test[j % cols])

    plt.tight_layout()
    plt.savefig(f"{output_path}/{title}_{data_type}_{model}_test_confusion_matrices_by_session.png")
    plt.close(fig_test)

    # Combined confusion matrix across all sessions
    combined_real_labels = [label for session in train_real_labels_all + test_real_labels_all for label in session]
    combined_pred_labels = [label for session in train_pred_labels_all + test_pred_labels_all for label in session]

    combined_cm = confusion_matrix(combined_real_labels, combined_pred_labels)

    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(combined_cm).plot(cmap='Purples', values_format='d')
    # plt.title(f'{title} - Combined Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_path}/{title}_{data_type}_{model}_combined_confusion_matrix.png")
    plt.close()


def create_radar_plot(sessions_list, train_acc_list, test_acc_list, train_chance_acc_list, test_chance_acc_list,
                      title, output_path, data_type, model):
    num_sessions = len(sessions_list)
    angles = np.linspace(0, 2 * np.pi, num_sessions, endpoint=False).tolist()
    angles += angles[:1]

    train_acc = train_acc_list + train_acc_list[:1]
    test_acc = test_acc_list + test_acc_list[:1]
    train_chance_acc = train_chance_acc_list + train_chance_acc_list[:1]
    test_chance_acc = test_chance_acc_list + test_chance_acc_list[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, train_acc, linewidth=2, color='blue', label='Train Accuracy')
    ax.plot(angles, test_acc, linewidth=2, color='green', label='Test Accuracy')
    ax.plot(angles, test_chance_acc, linewidth=2.5, linestyle='--', color='black', label='Chance Test Accuracy')

    for angle, value in zip(angles[:-1], train_acc[:-1]):
        ax.annotate(f'{value:.2f}',
                    xy=(angle, value),
                    xytext=(10, 10),
                    textcoords='offset points', ha='center', va='center', fontsize=9, color='blue')

    for angle, value in zip(angles[:-1], test_acc[:-1]):
        ax.annotate(f'{value:.2f}',
                    xy=(angle, value),
                    xytext=(10, -15),
                    textcoords='offset points', ha='center', va='center', fontsize=9, color='green')

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.2, 1.2, 0.2))
    ax.set_yticklabels([f'{v:.1f}' for v in np.arange(0.2, 1.2, 0.2)], fontsize=9, color='gray')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sessions_list, fontsize=10)
    # ax.set_title(f'{title} - Accuracy (Radar Plot) ({data_type})', fontsize=14, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_path}/{title}_{data_type}_{model}_radar_accuracy.png")
    plt.close()


def calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
                                test_pred_labels_all, sessions_list, title):
    try:
        plot_confusion_matrices(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
                                test_pred_labels_all,
                                sessions_list, title)
    except Exception as e:
        print(e)
    # Lists to store metrics for plotting
    bal_train_acc_list, train_acc_list, bal_train_f1_list, train_f1_list = [], [], [], []
    bal_test_acc_list, test_acc_list, bal_test_f1_list, test_f1_list = [], [], [], []
    train_chance_acc_list, test_chance_acc_list = [], []

    # Train metrics
    for i in range(len(train_real_labels_all)):
        train_real_labels = train_real_labels_all[i]
        train_pred_labels = train_pred_labels_all[i]

        bal_train_acc = balanced_accuracy_score(train_real_labels, train_pred_labels)
        train_acc = accuracy_score(train_real_labels, train_pred_labels)
        bal_train_f1_score = f1_score(train_real_labels, train_pred_labels, average='macro')
        train_f1_score = f1_score(train_real_labels, train_pred_labels, average='micro')

        # Append to lists
        bal_train_acc_list.append(bal_train_acc)
        train_acc_list.append(train_acc)
        bal_train_f1_list.append(bal_train_f1_score)
        train_f1_list.append(train_f1_score)

        label_counts = Counter(train_real_labels)
        most_frequent_label_count = label_counts.most_common(1)[0][1]
        chance_level_accuracy = most_frequent_label_count / len(train_real_labels)
        train_chance_acc_list.append(chance_level_accuracy)

        print(f"Title: {title}, Train Session: {sessions_list[i]}, CA: {chance_level_accuracy:.5f}, "
              f"BTA: {bal_train_acc:.5f}, BTF1: {bal_train_f1_score:.5f}, "
              f"TA: {train_acc:.5f}, TF1: {train_f1_score:.5f}")

    # Test metrics
    for i in range(len(test_real_labels_all)):
        test_real_labels = test_real_labels_all[i]
        test_pred_labels = test_pred_labels_all[i]

        bal_test_acc = balanced_accuracy_score(test_real_labels, test_pred_labels)
        test_acc = accuracy_score(test_real_labels, test_pred_labels)
        bal_test_f1_score = f1_score(test_real_labels, test_pred_labels, average='macro')
        test_f1_score = f1_score(test_real_labels, test_pred_labels, average='micro')

        # Append to lists
        bal_test_acc_list.append(bal_test_acc)
        test_acc_list.append(test_acc)
        bal_test_f1_list.append(bal_test_f1_score)
        test_f1_list.append(test_f1_score)

        label_counts = Counter(test_real_labels)
        most_frequent_label_count = label_counts.most_common(1)[0][1]
        chance_level_accuracy = most_frequent_label_count / len(test_real_labels)
        test_chance_acc_list.append(chance_level_accuracy)

        print(f"Title: {title}, Test Session: {sessions_list[i]}, CA: {chance_level_accuracy:.5f}, "
              f"BTA: {bal_test_acc:.5f}, BTF1: {bal_test_f1_score:.5f}, "
              f"TA: {test_acc:.5f}, TF1: {test_f1_score:.5f}")

    # Plot the session-wise metrics
    sessions = np.arange(1, len(sessions_list) + 1)
    for i in range(len(sessions_list)):
        sessions_list[i] = sessions_list[i][:9]

    # plt.figure(figsize=(12, 8))
    # plt.plot(sessions, train_acc_list, label='Train Accuracy', marker='o')
    # plt.plot(sessions, test_acc_list, label='Test Accuracy', marker='o')
    # plt.plot(sessions, train_chance_acc_list, label='Chance Train Accuracy', marker='x')
    # plt.plot(sessions, test_chance_acc_list, label='Chance Test Accuracy', marker='x')
    # plt.xlabel('Session')
    # plt.ylabel('Accuracy')
    # plt.title(f'{title} - Accuracy')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"{output_path}/{title}_{data_type}_{model}_accuracy.png")
    # plt.close()

    bar_width = 0.35  # Width for the train and test bars
    opacity = 0.8  # Opacity for the bar plots

    # Plotting
    plt.figure(figsize=(len(sessions_list) * 2, 10))
    fig, ax = plt.subplots(figsize=(len(sessions_list) * 2, 10))

    # Train and Test Accuracy Bars
    train_bars = ax.bar(sessions - bar_width / 2, train_acc_list, bar_width,
                        alpha=opacity, color='blue', label='Train Accuracy')
    test_bars = ax.bar(sessions + bar_width / 2, test_acc_list, bar_width,
                       alpha=opacity, color='green', label='Test Accuracy')

    # Chance Accuracy Bars (with outline only)
    chance_train_bars = ax.bar(sessions - bar_width / 2, train_chance_acc_list, bar_width,
                               facecolor='none', edgecolor='black', linewidth=2.5, linestyle='--',
                               label='Chance Train Accuracy')
    chance_test_bars = ax.bar(sessions + bar_width / 2, test_chance_acc_list, bar_width,
                              facecolor='none', edgecolor='black', linewidth=2.5, linestyle='--',
                              label='Chance Test Accuracy')

    # Annotate Train and Test bars with their values
    for bar in train_bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text position
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)

    for bar in test_bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text position
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Set labels and title
    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    # ax.set_title(f'{title} - Accuracy', fontsize=14)
    ax.set_xticks(sessions)
    ax.set_xticklabels(sessions_list)
    ax.legend()

    # Display the grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(f"{output_path}/{title}_{data_type}_{model}_bar_accuracy.png")
    plt.close()

    plt.figure(figsize=(len(sessions_list) * 2, 8))
    plt.plot(sessions, bal_train_acc_list, label='Balanced Train Accuracy', marker='o')
    plt.plot(sessions, bal_test_acc_list, label='Balanced Test Accuracy', marker='o')
    plt.xlabel('Session')
    plt.ylabel('Accuracy')
    # plt.title(f'{title} - Balanced Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/{title}_{data_type}_{model}_balanced_accuracy.png")
    plt.close()

    plt.figure(figsize=(len(sessions_list) * 2, 8))
    plt.plot(sessions, bal_train_f1_list, label='Balanced Train F1', marker='x')
    plt.plot(sessions, train_f1_list, label='Train F1', marker='o')
    plt.plot(sessions, bal_test_f1_list, label='Balanced Test F1', marker='x')
    plt.plot(sessions, test_f1_list, label='Test F1', marker='o')
    plt.xlabel('Session')
    plt.ylabel('F1 Score')
    # plt.title(f'{title} - F1 Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/{title}_{data_type}_{model}_f1_scores.png")
    plt.close()

    # Calculate combined values across all sessions
    combined_bal_train_acc = np.mean(bal_train_acc_list)
    std_bal_train_acc = np.std(bal_train_acc_list)

    combined_train_acc = np.mean(train_acc_list)
    std_train_acc = np.std(train_acc_list)

    combined_bal_train_f1 = np.mean(bal_train_f1_list)
    std_bal_train_f1 = np.std(bal_train_f1_list)

    combined_train_f1 = np.mean(train_f1_list)
    std_train_f1 = np.std(train_f1_list)

    combined_train_chance_acc = np.mean(train_chance_acc_list)
    std_train_chance_acc = np.std(train_chance_acc_list)

    combined_bal_test_acc = np.mean(bal_test_acc_list)
    std_bal_test_acc = np.std(bal_test_acc_list)

    combined_test_acc = np.mean(test_acc_list)
    std_test_acc = np.std(test_acc_list)

    combined_bal_test_f1 = np.mean(bal_test_f1_list)
    std_bal_test_f1 = np.std(bal_test_f1_list)

    combined_test_f1 = np.mean(test_f1_list)
    std_test_f1 = np.std(test_f1_list)

    combined_test_chance_acc = np.mean(test_chance_acc_list)
    std_test_chance_acc = np.std(test_chance_acc_list)

    create_radar_plot(sessions_list, train_acc_list, test_acc_list, train_chance_acc_list,
                      test_chance_acc_list, title, output_path, data_type, model)

    print(f"\nCombined Metrics (Mean ± Std):")
    print(f"Combined Train - BTA: {combined_bal_train_acc:.5f} ± {std_bal_train_acc:.5f}, "
          f"TA: {combined_train_acc:.5f} ± {std_train_acc:.5f}, "
          f"BTF1: {combined_bal_train_f1:.5f} ± {std_bal_train_f1:.5f}, "
          f"TF1: {combined_train_f1:.5f} ± {std_train_f1:.5f}, "
          f"CA: {combined_train_chance_acc:.5f} ± {std_train_chance_acc:.5f}")

    print(f"Combined Test - BTA: {combined_bal_test_acc:.5f} ± {std_bal_test_acc:.5f}, "
          f"TA: {combined_test_acc:.5f} ± {std_test_acc:.5f}, "
          f"BTF1: {combined_bal_test_f1:.5f} ± {std_bal_test_f1:.5f}, "
          f"TF1: {combined_test_f1:.5f} ± {std_test_f1:.5f}, "
          f"CA: {combined_test_chance_acc:.5f} ± {std_test_chance_acc:.5f}")


# Enter the path to the targeted input data
def arg_parser():
    parser = argparse.ArgumentParser(description='MLP baseline')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen, ibl or Neuronexus', default='Allen')
    parser.add_argument('--data_type', type=str, help='Type of data to use: raw or spectrogram', default='spectrogram')
    parser.add_argument('--model', type=str, help='Model to use: mlp, linear, SimCLR_MLP or SimCLR_LR', default='mlp')
    parser.add_argument('--subcategory', type=str, help='Subcategory of model: separate or end_to_end', default='')
    return parser.parse_args()


args = arg_parser()
data, data_type, model, subcategory = args.data, args.data_type, args.model, args.subcategory
print(f"Data: {data}, Data Type: {data_type}, Model: {model}")

if data == 'Allen':
    sessions = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
elif data == 'ibl':
    sessions = ['15763234-d21e-491f-a01b-1238eb96d389', '1a507308-c63a-4e02-8f32-3239a07dc578',
                '4a45c8ba-db6f-4f11-9403-56e06a33dfa4', '56956777-dca5-468c-87cb-78150432cc57',
                '5b49aca6-a6f4-4075-931a-617ad64c219c', '73918ae1-e4fd-4c18-b132-00cb555b1ad2',
                'b39752db-abdb-47ab-ae78-e8608bbf50ed']
elif data == "Neuronexus":
    sessions = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]

pickle_path = f'../results/{data}/'
output_path = f'../results/{data}/{data_type}/{model}/'
if subcategory != '':
    output_path = f'../results/{data}/{data_type}/{model}/{subcategory}/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

if __name__ == '__main__':
    if model != 'SimCLR_MLP' and model != 'SimCLR_LR':
        try:
            if args.data == 'Allen':
                index_of_interest = [1, 37, 29, 39, 25, 24, 23]
            elif args.data == 'ibl':
                index_of_interest = [0, 7, 8, 3, 5, 6, 1]
            with open(f"{pickle_path}{data_type}/{model}/within_train_real_labels.pickle", 'rb') as handle:
                train_real_labels_all = pickle.load(handle)
                if len(train_real_labels_all) > len(sessions):
                    train_real_labels_all = [train_real_labels_all[i] for i in index_of_interest]
            with open(f"{pickle_path}{data_type}/{model}/within_train_pred_labels.pickle", 'rb') as handle:
                train_pred_labels_all = pickle.load(handle)
                if len(train_pred_labels_all) > len(sessions):
                    train_pred_labels_all = [train_pred_labels_all[i] for i in index_of_interest]
            with open(f"{pickle_path}{data_type}/{model}/within_test_real_labels.pickle", 'rb') as handle:
                test_real_labels_all = pickle.load(handle)
                if len(test_real_labels_all) > len(sessions):
                    test_real_labels_all = [test_real_labels_all[i] for i in index_of_interest]
            with open(f"{pickle_path}{data_type}/{model}/within_test_pred_labels.pickle", 'rb') as handle:
                test_pred_labels_all = pickle.load(handle)
                if len(test_pred_labels_all) > len(sessions):
                    test_pred_labels_all = [test_pred_labels_all[i] for i in index_of_interest]

            calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
                                        test_pred_labels_all, sessions, f"Within Session for {data} "
                                                                        f"using {model} {subcategory}")
        except Exception as e:
            print(f"Error in loading within session data: {e}")
            print("Please check the pickle files for the within session data.")
        # change to across
        # with open(f"{pickle_path}{data_type}/{model}/inductive_train_real_labels.pickle", 'rb') as handle:
        #     train_real_labels_all = pickle.load(handle)
        # with open(f"{pickle_path}{data_type}/{model}/inductive_train_pred_labels.pickle", 'rb') as handle:
        #     train_pred_labels_all = pickle.load(handle)
        # with open(f"{pickle_path}{data_type}/{model}/inductive_test_real_labels.pickle", 'rb') as handle:
        #     test_real_labels_all = pickle.load(handle)
        # with open(f"{pickle_path}{data_type}/{model}/inductive_test_pred_labels.pickle", 'rb') as handle:
        #     test_pred_labels_all = pickle.load(handle)
        # calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
        #                             test_pred_labels_all, sessions, "Inductive Session")

        try:
            with open(f"{pickle_path}{data_type}/{model}/inductive_train_real_labels.pickle", 'rb') as handle:
                train_real_labels_all = pickle.load(handle)
                if len(train_real_labels_all) > len(sessions):
                    train_real_labels_all = [train_real_labels_all[i] for i in index_of_interest]
            with open(f"{pickle_path}{data_type}/{model}/inductive_train_pred_labels.pickle", 'rb') as handle:
                train_pred_labels_all = pickle.load(handle)
                if len(train_pred_labels_all) > len(sessions):
                    train_pred_labels_all = [train_pred_labels_all[i] for i in index_of_interest]
            with open(f"{pickle_path}{data_type}/{model}/inductive_test_real_labels.pickle", 'rb') as handle:
                test_real_labels_all = pickle.load(handle)
                if len(test_real_labels_all) > len(sessions):
                    test_real_labels_all = [test_real_labels_all[i] for i in index_of_interest]
            with open(f"{pickle_path}{data_type}/{model}/inductive_test_pred_labels.pickle", 'rb') as handle:
                test_pred_labels_all = pickle.load(handle)
                if len(test_pred_labels_all) > len(sessions):
                    test_pred_labels_all = [test_pred_labels_all[i] for i in index_of_interest]

            calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
                                        test_pred_labels_all, sessions, f"Inductive Across Session for {data} "
                                                                        f"using {model} {subcategory}")
        except Exception as e:
            print(f"Error in loading across session data: {e}")
            print("Please check the pickle files for the across session data.")
    else:
        train_real_labels_all, train_pred_labels_all, test_real_labels_all, test_pred_labels_all = [], [], [], []
        final_sessions = []
        for session in sessions:
            try:
                temp = f"{pickle_path}{data_type}/{model}/{subcategory}/{session}_within_session.pickle"
                with open(f"{pickle_path}{data_type}/{model}/{subcategory}/{session}_within_session.pickle",
                          'rb') as handle:
                    saved_data = pickle.load(handle)
                    saved_train_real_labels = saved_data['train_real_labels']
                    saved_train_pred_labels = saved_data['train_pred_labels']
                    saved_test_real_labels = saved_data['test_real_labels']
                    saved_test_pred_labels = saved_data['test_pred_labels']
                    train_real_labels_all.append(saved_train_real_labels)
                    train_pred_labels_all.append(saved_train_pred_labels)
                    test_real_labels_all.append(saved_test_real_labels)
                    test_pred_labels_all.append(saved_test_pred_labels)
                    final_sessions.append(session)
            except:
                print(f"Session {session} not found")
                continue
        calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
                                    test_pred_labels_all, final_sessions, f"Within Session for {data} "
                                                                          f"using {model} {subcategory}")

        train_real_labels_all, train_pred_labels_all, test_real_labels_all, test_pred_labels_all = [], [], [], []
        final_sessions = []
        for session in sessions:
            try:
                with open(f"{pickle_path}{data_type}/{model}/{subcategory}/{session}_across_session.pickle",
                          'rb') as handle:
                    saved_data = pickle.load(handle)
                    saved_train_real_labels = saved_data['train_real_labels']
                    saved_train_pred_labels = saved_data['train_pred_labels']
                    saved_test_real_labels = saved_data['test_real_labels']
                    saved_test_pred_labels = saved_data['test_pred_labels']
                    train_real_labels_all.append(saved_train_real_labels)
                    train_pred_labels_all.append(saved_train_pred_labels)
                    test_real_labels_all.append(saved_test_real_labels)
                    test_pred_labels_all.append(saved_test_pred_labels)
                    final_sessions.append(session)
            except:
                print(f"Session {session} not found")
                continue
        calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
                                    test_pred_labels_all, final_sessions, f"Inductive Across Session for {data} "
                                                                          f"using {model} {subcategory}")
