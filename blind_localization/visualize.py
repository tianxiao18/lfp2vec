import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from sklearn.decomposition import PCA
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from functools import partial
import textwrap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import torch
import plotly.graph_objs as go
import os
import networkx as nx

def visualize_raw_signals(raw_signal, channel_region_map, skipped_channels, prediction=None, indices=None, T = 1000, sr = 20000, saved_path='raw_signal.png'):
    """
    Visualize raw signals colored by region (skipping bad channels)
    if prediction is None, color represents ground truth labels
    otherwise, color represents the train/val/test prediction
    """
    offset = 5
    sr_converted = int(sr/1000) #samples/ms
    t_offset = T + 100

    time = np.arange(0, T, 1/sr_converted)

    colors = ["red", "orange", "green", "blue", "magenta", "black"]
    regions = ["cortex", "CA1", "CA2", "CA3", "DG", "UNK"]
    regions_to_colors = dict(zip(regions, colors))
    
    skip_arr = np.zeros(len(raw_signal))
    skip_arr[skipped_channels] = 1

    if prediction:
        y_pred_train, y_pred_val, y_pred_test = prediction
        channel_idx_train, channel_idx_val, channel_idx_test = indices

    plt.figure(figsize=(18, 16))
    for shank in range(8):
        for i in range(128): 
            channel = shank*128+i

            row = channel_region_map[channel_region_map["channels"] == channel]
            region = row.iloc[0]["regions"] if len(row) > 0 else "UNK"

            if not skip_arr[channel]:
                if not prediction:
                    color = regions_to_colors[region]

                else:
                    if channel in channel_idx_val: 
                        r = int(y_pred_val[channel_idx_val == channel][0])
                        color = regions_to_colors[regions[r]]
                    elif channel in channel_idx_train:
                        r = int(y_pred_train[channel_idx_train == channel][0])
                        color = regions_to_colors[regions[r]]
                    elif channel in channel_idx_test:
                        r = int(y_pred_test[channel_idx_test == channel][0])
                        color = regions_to_colors[regions[r]]
                    else:
                        print(channel)
                        color = "black"
            
                plt.plot(time+t_offset*shank, raw_signal[channel, :T*sr_converted]+offset*i, c=color)

    legend_handles = [Line2D([0], [0], color=color, lw=2, label=list(regions_to_colors.keys())[i]) for i, color in enumerate(colors)]

    # plt.legend(handles=legend_handles)

    plt.xlabel("time(ms)")
    plt.ylabel("channel")
    plt.title("Raw signal detected by 8 shanks in 1000 ms")
    plt.show()
    plt.savefig(saved_path) if saved_path else None


def visualize_channel_features(channel_features):
    fig, axes = plt.subplots(8, 1, figsize=(21, 9.6))

    for shank, ax in enumerate(axes.flat):
        im = ax.imshow(channel_features[shank*128:shank*128+128, :-1].T)
        ax.set_yticks(np.arange(5), ["cortex", "CA1", "CA2", "CA3" ,"DG"])
        ax.set_xticks(np.arange(0, 128, 10), np.arange(shank*128, shank*128+128, 10))

    title_obj = plt.suptitle("Similarity between channel to each brain region", fontsize=15)
    title_pos = title_obj.get_position()
    title_obj.set_position((title_pos[0] - 0.05, title_pos[1]))
    plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()


def region_accuracy_score(y_true, y_pred):
    accuracy_per_region = []
    regions = ["cortex", "CA1", "CA2", "CA3", "DG"]

    for r in range(len(regions)):
        if len(y_true[y_true == r]) > 0:
            acc = accuracy_score(y_true[y_true == r], y_pred[y_true == r])
        else:
            acc = 0

        accuracy_per_region.append(acc)

    return accuracy_per_region


def visualize_accuracy(y_train, y_val, y_pred_train, y_pred, alignment=False, ax=None):
    accuracy_per_region_train = []
    accuracy_per_region_val= []
    regions = ["cortex", "CA1", "CA2", "CA3", "DG"]

    for r in range(5):
        if len(y_train[y_train == r]) > 0:
            train_acc = accuracy_score(y_train[y_train == r], y_pred_train[y_train == r])
            val_acc = accuracy_score(y_val[y_val == r], y_pred[y_val == r])
        else:
            train_acc, val_acc = 0, 0

        accuracy_per_region_train.append(train_acc)
        accuracy_per_region_val.append(val_acc)

    bar_width=0.2
    index = np.arange(len(regions))
    ax_flag = False

    if ax is None:
        fig, ax = plt.subplots()
        ax_flag = True

    label = 'Train' if not alignment else "Alignment"
    ax.bar(index, accuracy_per_region_train, bar_width, label=label)
    ax.bar(index + bar_width, accuracy_per_region_val, bar_width, label='Validation')

    for r in range(5):
        count_train = len(y_train[y_train == r])
        count_val = len(y_val[y_val == r])
        ax.text(r, accuracy_per_region_train[r], str(count_train), ha='center', va='bottom')
        ax.text(r+bar_width, accuracy_per_region_val[r], str(count_val), ha='center', va='bottom')

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(regions)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_ylabel("Accuracy")
    ax.set_title("Region classification accuracy")
    ax.legend(loc='lower left')
    if ax_flag:
        plt.show()
    return ax

def visualize_confusion_matrix(y_true, y_pred, title='Validation Confusion Plot', savepath=None):
    c_matrix = confusion_matrix(y_true, y_pred, labels=range(5))
    regions = ["cortex", "CA1", "CA2", "CA3", "DG"]

    plt.figure()
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=regions, yticklabels=regions)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.savefig(savepath) if savepath else None


def animate_features(channel_feature_across_time, frames):
    fig, axes = plt.subplots(8, 1, figsize=(21, 9.6))
    images = []

    # Initialize subplots with zeros (or any initial image data)
    for i in range(8):
        image = axes[i].imshow(np.zeros((100, 100)), animated=True)
        images.append(image)

    def init():
        # Initialize each subplot with zeros (or any initial image data)
        for image in images:
            image.set_array(np.zeros((100, 100)))
        return images

    def update(frame):
        channel_features = channel_feature_across_time[frame]
        
        for shank, ax in enumerate(axes.flat):
            im = ax.imshow(channel_features[shank*128:shank*128+128, :-1].T)
            ax.set_yticks(np.arange(5), ["cortex", "CA1", "CA2", "CA3" ,"DG"])
            ax.set_xticks(np.arange(0, 128, 10), np.arange(shank*128, shank*128+128, 10))
            images[shank].set_array(im.get_array())

        return images

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    plt.close()
    return HTML(ani.to_jshtml())


def animate_tensor(custom_update, tensor, frames, n_rows, n_cols, titles=None, width=20, height=4, vmin=0, vmax=1):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))
    images = []

    if n_rows == 1 or n_cols == 1:
        axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            if titles:
                axes[i][j].set_title(titles[i][j])
            image = axes[i][j].imshow(np.zeros((tensor.shape[3], tensor.shape[4])), vmin=vmin, vmax=vmax, animated=True)
            images.append(image)

    def init():
        for image in images:
            image.set_array(np.zeros((tensor.shape[3], tensor.shape[4])))
        return images

    update = partial(custom_update, axes=axes, images=images, tensor=tensor)

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    plt.close()
    return HTML(ani.to_jshtml())


def animate(data, custom_plot_func, frames):
    fig, ax = plt.subplots(figsize=(8, 6))

    def init():
        return ax,

    def update(frame):
        frame_data = data[frame]
        custom_plot_func(ax, frame_data)
        return ax,

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
    
    plt.close(fig)  # Prevents duplicate display of static image
    return HTML(ani.to_jshtml())


def visualize_alignment(channel_features, channel_ridx_map, channel_features_new, channel_ridx_map_new, channel_features_transformed):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    xcorr = channel_features[channel_ridx_map[:, 0]][:, :-1]
    
    regions = ["cortex", "CA1", "CA2", "CA3", "DG"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    pca = PCA(n_components=2)
    pca.fit(xcorr)

    visualize_aligned_features(channel_features, channel_ridx_map, channel_features_new, 
                               channel_ridx_map_new, regions, colors, pca, title="Before Alignment", ax=axes[0])
    visualize_aligned_features(channel_features, channel_ridx_map, channel_features_transformed,
                               channel_ridx_map_new, regions, colors, pca, title="After Alignment",ax=axes[1])
    plt.show()


def visualize_aligned_features(channel_features, channel_ridx_map, channel_features_new, channel_ridx_map_new,
                            regions, colors, pca, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for r in range(5):
        xcorr_new = channel_features_new[channel_ridx_map_new[:, 0]][channel_ridx_map_new[:, 1] == r][:, :-1] 
        if len(xcorr_new) > 0:
            data_pca = pca.transform(xcorr_new)
            ax.scatter(data_pca[:, 0], data_pca[:, 1], marker='x', color=colors[r], zorder=1, s=20)

        xcorr = channel_features[channel_ridx_map[:, 0]][channel_ridx_map[:, 1] == r][:, :-1]
        if len(xcorr) > 0:
            data_pca = pca.transform(xcorr)
            ax.scatter(data_pca[:, 0], data_pca[:, 1], marker='o', label=regions[r], color=colors[r],zorder=1, s=20)
    
    ax.set_title(title)
    ax.legend()

    return ax


def visualize_features_by_regions(ax, channel_features, r, element_width = 10, element_height = 0.5):

    concat_features = np.concatenate([channel_features[s*128:(s+1)*128, r:r+1] for s in range(8)], axis=1)

    if concat_features.max() != 0 and concat_features.max() != 0:
        normalized_features = (concat_features - concat_features.min()) / (concat_features.max() - concat_features.min())
    else:
        normalized_features = np.zeros_like(concat_features)

    rgba_color = plt.cm.viridis(normalized_features)

    for i in range(concat_features.shape[0]):
        for j in range(concat_features.shape[1]):
            rect = patches.Rectangle((j*element_width, i*element_height), element_width, element_height,
                                    linewidth=1, edgecolor='none', facecolor=rgba_color[i][j])
            ax.add_patch(rect)

    ax.set_xlim([0, element_width * concat_features.shape[1]])
    ax.set_ylim([0, element_height * concat_features.shape[0]])

    ax.set_xticks(np.arange(8)*element_width, np.arange(8), fontsize=12)
    ax.set_yticks(np.arange(0, 128, 10)*element_height, np.arange(0, 128, 10), fontsize=12)

    return ax


def visualize_features_dataset(channel_features_ls):
    n_figures = len(channel_features_ls)
    n_regions = len(channel_features_ls[0][0])

    session_names = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    regions = ["cortex", "CA1", "CA2", "CA3", "DG"]
    
    fig, axs = plt.subplots(n_regions, n_figures, figsize=(4*n_figures, 3*n_regions))

    for fig_num in range(n_figures):
        channel_features = channel_features_ls[fig_num]

        for region_index in range(n_regions):
            ax = axs[region_index, fig_num]
            visualize_features_by_regions(ax, channel_features, region_index)

            if region_index == 0:
                axs[region_index, fig_num].set_title(session_names[fig_num], fontsize=20)
            
            if fig_num == 0:
                axs[region_index, fig_num].set_ylabel(regions[region_index], fontsize=20)
        
    plt.tight_layout()
    plt.show()


def visualize_features_3d(session_names, reduced_signal_ls, channel_labels_ls, height=120, width=120, marker_size=2):
    n_rows = 1
    n_cols = len(session_names)
    df_ls = []
    
    for i, session_name in enumerate(session_names):
        channel_labels = channel_labels_ls[i]
        labels = np.argmax(channel_labels, axis=1) + 1
        channel_labels = channel_labels * labels[:, np.newaxis]

        class_labels = np.sum(channel_labels, axis=1).astype(int)
        colors = ["lightgray", "red", "orange", "green", "blue", "purple"]
        channel_colors = [colors[j] for j in class_labels]

        df = pd.DataFrame(reduced_signal_ls[i], columns=['x', 'y', 'z'])
        df['color'] = channel_colors
        df['session_name'] = session_name
        df_ls.append(df)

    fig = make_subplots(rows=n_rows, cols=n_cols,subplot_titles=tuple(session_names),
                        specs=[[{'type':'scatter3d'} for c in range(n_cols)] for r in range(n_rows)])

    for r in range(n_rows):
        for c in range(n_cols):
            df = df_ls[c]
            scatter = px.scatter_3d(df, x='x', y='y', z='z', color='color', color_discrete_sequence=df["color"].unique())
            for trace in scatter.data:
                fig.add_trace(trace, row=r+1, col=c+1)

    fig.update_traces(marker=dict(size=marker_size), showlegend=False, selector=dict(type='scatter3d'))

    fig.update_layout(height=height, width=width*len(session_names),margin=dict(l=0, r=0, b=0, t=0))
                    
    for i in range(1, n_cols + 1):
        fig.update_layout(**{
            f'scene{i}': dict(
                xaxis=dict(showgrid=True, gridcolor='lightgray', backgroundcolor='white',showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor='lightgray', backgroundcolor='white',showticklabels=False),
                zaxis=dict(showgrid=True, gridcolor='lightgray', backgroundcolor='white',showticklabels=False),
                xaxis_title="", yaxis_title="", zaxis_title="",aspectmode='cube'
            )
        })

    fig.show()


def visualize_confusion_matrix_all(y_target_ls, y_pred_ls):
    n_session = int(np.sqrt(len(y_pred_ls)))
    fig, axes = plt.subplots(n_session, n_session, figsize=(3*n_session, 3*n_session))

    for i in range(n_session):
        for j in range(n_session):
            idx = i*n_session+j
            y_true = y_target_ls[idx]
            y_pred = y_pred_ls[idx]

            c_matrix = confusion_matrix(y_true, y_pred, labels=range(5))
            sns.heatmap(c_matrix, ax=axes[i][j], annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=False, yticklabels=False)

    plt.show()

def visualize_accuracy_all(accuracy_ls):
    n_sessions = int(np.sqrt(len(accuracy_ls)))
    sns.heatmap(np.array(accuracy_ls).reshape(n_sessions, n_sessions), cmap='Blues', annot=True)
    plt.show()

def visualize_sequence(data_list):
    fig = plt.figure()

    for i, (X_embed, Y_embed) in enumerate(data_list):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.scatter(X_embed[:, 0], X_embed[:, 1], X_embed[:, 2], c=Y_embed, cmap='viridis', marker='o')

    plt.show()

def visualize_correlation(channel_features_raw, channel_features):
    # from scipy.spatial.distance import pdist, squareform
    # from scipy.cluster.hierarchy import linkage, leaves_list

    # dist_matrix = pdist(channel_features_raw, 'euclidean')
    # linkage_matrix = linkage(dist_matrix, method='ward')
    # ordered_indices = leaves_list(linkage_matrix)
    # channel_features_raw = channel_features_raw[:, ordered_indices][ordered_indices, :]

    # dist_matrix = pdist(channel_features, 'euclidean')
    # linkage_matrix = linkage(dist_matrix, method='ward')
    # ordered_indices = leaves_list(linkage_matrix)
    # channel_features = channel_features[:, ordered_indices][ordered_indices, :]

    index_order = np.arange(1024).reshape(8, 128).swapaxes(0, 1).flatten()

    plt.figure(figsize=(6, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(channel_features_raw[index_order, :][:, index_order])
    plt.subplot(1, 2, 2)
    plt.imshow(channel_features[index_order, :][:, index_order])
    plt.show()


def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)

    plt.plot([], c=color_code, label=label)
    plt.legend()


def visualize_box_plot(model1_accs, model2_accs, model_names=["model1", "model2"]):
    plt.rcParams["figure.figsize"] = (14,5)
    plt.rcParams.update({'font.size': 12})

    ticks = ["cortex", "CA1", "CA2", "CA3", "DG"]

    model1_acc_plot = plt.boxplot(model1_accs,
                                positions=np.array(np.arange(len(model1_accs)))*2.0-0.1,
                                widths=0.1, showfliers=True)
    model2_acc_plot = plt.boxplot(model2_accs,
                                positions=np.array(np.arange(len(model2_accs)))*2.0+0.1,
                                widths=0.1, showfliers=True)

    define_box_properties(model1_acc_plot, '#000000', model_names[0])
    define_box_properties(model2_acc_plot, '#D7191C', model_names[1])

    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.title('Decoding Performance Comparison')
    plt.show()


def visualize_acc_with_data_size(n_trials, cv_accuracy_scores):
    plt.scatter(n_trials, cv_accuracy_scores)
    plt.plot(n_trials, cv_accuracy_scores)
    plt.xlabel("n trials")
    plt.ylabel("accuarcy")
    plt.title("Model accuracy vs. trials")
    plt.show()


def visualize_session_accuracy(train_accuracy, test_accuracy, session_names, labels=['Train', 'Test'], title='session_accuracy'):
    index = np.arange(len(train_accuracy))

    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.2

    ax.bar(index, train_accuracy, bar_width, color='#FFB000', label=labels[0])
    ax.bar(index + bar_width, test_accuracy, bar_width, color='#FF8032', label=labels[1])

    labels = [textwrap.fill(session, width=6) for session in session_names]
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel("Accuracy")
    ax.set_title("Region classification accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.show()


def visualize_data_pca(channel_features_all, channel_labels_all):
    X = np.vstack(channel_features_all)
    y = np.hstack(channel_labels_all)
    pca = PCA(n_components=3)
    x_hat = pca.fit_transform(X)

    colors = ["red", "orange", "green", "blue", "purple", "lightgray"]
    channel_colors = [colors[j] for j in y]

    ax = plt.subplot(111, projection='3d')
    ax.scatter(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], c=channel_colors)
    plt.show()

def visualize_train_losses(training_losses, validation_losses, labels=['train', 'validaiton']):
    plt.plot(training_losses, label=labels[0])
    plt.plot(validation_losses, label=labels[1])
    plt.title("Model Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

def visualize_time_varying_predictions(raw_signal, prediction, indices, offset=5, t_offset=1100):
    # raw_signal of shape n_trials, n_channels, n_features
    n_trials, n_channels, n_features = raw_signal.shape
    time = np.arange(0, 0.5, n_features)

    plt.figure(figsize=(18, 16))
    colors = ["red", "orange", "green", "blue", "magenta", "black"]

    for t in range(n_trials):
        for c in range(n_channels): 
            # label = labels[c]
            plt.plot(time+t_offset*t, raw_signal[t][c]+offset*c)
    plt.show()

def visualize_embeddings(model, dataloaders, device, title='before', vis_dim="3d", color_map=None):
    model.eval()
    embeddings = []
    labels = []
    sessions = []

    for dataloader in dataloaders:
        with torch.no_grad():
            for data, label, (session, trial_idx) in dataloader:
                trial_indices = trial_idx == 0

                if trial_indices.any():
                    spectrograms, _ = data
                    spectrograms = spectrograms[trial_indices].to(device)
                    label = label[trial_indices].to('cpu')
                    session_filtered = session[trial_indices].to('cpu')

                    z = model(spectrograms)
                    embedding = z.cpu().numpy()
                    embeddings.append(embedding)
                    labels.append(label.numpy())
                    sessions.append(session_filtered.numpy())

    print(len(sessions), len(labels))    
    embeddings = np.concatenate(embeddings, axis=0)
    sessions = np.concatenate(sessions, axis=0)
    labels = np.concatenate(labels, axis=0)

    if vis_dim == "3d":
        visualize_embedding(embeddings, labels, title, sessions)
    else:
        label_color_map = {0: 'red', 1: 'orange', 2:'green', 3:'blue', 4:'magenta'}
        colors = np.array([label_color_map[label] for label in labels])
        
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors)  
        plt.colorbar(scatter)
        plt.title('t-SNE Projection of Contrastive Embeddings')
        plt.savefig(f'{title}_tsne.png')
        plt.close()

        umap_model = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = umap_model.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=5)
        plt.colorbar(scatter)
        plt.title('UMAP Projection of Contrastive Embeddings')
        plt.savefig(f'{title}_umap.png')
        plt.close()

        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=5)
        plt.colorbar(scatter)
        plt.title('PCA Projection of Contrastive Embeddings')
        plt.savefig(f'{title}_pca.png')
        plt.close()


def plot_embedding(data, labels, file_path, session_labels):
    print(file_path)
    # Define different markers for different sessions
    marker_symbols = ['circle', 'circle-open', 'cross', 'diamond',
            'diamond-open', 'square', 'square-open', 'x']
    session_marker_map = {session: marker_symbols[i % len(marker_symbols)] for i, session in enumerate(set(session_labels))}
    color_map = {0: 'red', 1: 'orange', 2: 'green', 3: 'blue', 4: 'purple'}
    acronyms_arr = ['Cortex', 'CA1', 'CA2', 'CA3', 'DG']

    # Create traces for each session
    traces = []
    for session in set(session_labels):
        session_indices = [i for i, s in enumerate(session_labels) if s == session]
        session_data = data[session_indices]
        session_labels_subset = [labels[i] for i in session_indices]
        
        trace = go.Scatter3d(
            x=session_data[:, 0],
            y=session_data[:, 1],
            z=session_data[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=[color_map[label] for label in session_labels_subset],  # Color points based on labels
                symbol=session_marker_map[session],
                opacity=0.8
            ),
            name=f'Session {session}'
        )
        traces.append(trace)
    for label, color in color_map.items():
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None],  # No actual data points, just a marker for the legend
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                symbol='circle',
            ),
            name=f'{acronyms_arr[label]}',  # Display color: label name
            showlegend=True
        ))
    # Define layout
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        )
    )
    
    # Create figure and add traces
    fig = go.Figure(data=traces, layout=layout)
    
    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write figure to HTML file
    fig.write_html(file_path)


def visualize_embedding(data, labels, session, session_labels):
    # 3D PCA
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data)
    file_path = f"/scratch/th3129/region_decoding/results/{session}_PCA.html"
    plot_embedding(pca_data, labels, file_path, session_labels)

def visualize_explained_variance(all_embeddings):
    pca = PCA()
    pca.fit(all_embeddings)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
    plt.legend()
    plt.savefig('explained_variance.png')

    non_zero_count = np.count_nonzero(all_embeddings, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(all_embeddings.shape[1]), non_zero_count, color='salmon')
    plt.xlabel('Dimension')
    plt.ylabel('Non-Zero Count')
    plt.title('Non-Zero Count per Dimension')
    plt.savefig('zero_count.png')

def visualize_graph(data, device, title="augmented_graph1"):
    edge_index = data.edge_index.to(device)
    pos = data.pos.to('cpu')
    labels = data.y.to('cpu')

    edges = edge_index.t().tolist()

    G = nx.DiGraph() 
    G.add_edges_from(edges)
    colors = ["red", "orange", "green", "blue", "magenta"]
    node_colors = [colors[labels[i]] for i in G.nodes]

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=75, edge_color='gray', font_size=15, font_weight='bold')
    plt.title('Graph Visualization from edge_index')
    plt.savefig(f'{title}.png')

def visualize_spectrograms(spectrograms, n_freq_bins, n_time_bins, name, sample_rate=20000, n_fft=2048):
    spectrogram = spectrograms[0][0].reshape((n_freq_bins, n_time_bins))
    n_features = n_freq_bins*n_time_bins
    
    frequencies = np.linspace(0, sample_rate/2, n_fft//2+1)[:n_freq_bins]
    time_steps = np.linspace(0, n_features/sample_rate, n_time_bins)

    plt.figure()
    plt.pcolormesh(time_steps, frequencies, spectrogram, shading='gouraud')
    plt.ylim(0, 500)
    plt.colorbar(label='Intensity [dB]')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(f'results/{name}_spectrogram.png')

def visualize_session_metrics(accuracy_metrics, session_names, labels=None, colors=None, title='session_accuracy', save=True):
    if labels and len(accuracy_metrics) != len(labels):
        raise ValueError("Number of accuracy metrics must match the number of labels.")

    if labels is None:
        labels = [f"Metric {i+1}" for i in range(len(accuracy_metrics))]

    num_metrics = len(accuracy_metrics)
    num_sessions = len(session_names)

    index = np.arange(num_sessions)
    bar_width = 0.8 / num_metrics

    default_colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#56B4E9', '#CC79A7', '#E69F00']
    
    if colors is None:
        colors = default_colors
        colors = (colors * (len(accuracy_metrics) // len(colors) + 1))[:len(accuracy_metrics)]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label) in enumerate(zip(accuracy_metrics, labels)):
        ax.bar(index + i * bar_width, metric, bar_width, label=label, color=colors[i])

    wrapped_labels = [textwrap.fill(session, width=6) for session in session_names]
    ax.set_xticks(index + (num_metrics - 1) * bar_width / 2)
    ax.set_xticklabels(wrapped_labels)

    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(f'{title}.png')
    plt.show()