import plotly.graph_objs as go
import argparse

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from script.utils import *

pickle_path = "./data"  # Define pickle data path
sessions_list = ["session1", "session2"]  # Define sessions list
data_type = "spectrogram"  # Define data type, either "spectrogram" or "raw"

def arg_parser():
    parser = argparse.ArgumentParser(description='MLP baseline')
    parser.add_argument('--data', type=str, help='Dataset to use: Allen or ibl', default='ibl')
    parser.add_argument('--data_type', type=str, help='Type of data to use: raw or spectrogram', default='spectrogram')
    return parser.parse_args()

args = arg_parser()
# data, data_type = args.data, args.data_type
print(f"Data: {args.data}, Data Type: {args.data_type}")

output_path = f"../figs/{args.data}/{args.data_type}"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# color map
hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Cortex'}
acronyms_arr = np.array(sorted(list(hc_acronyms)))
acronyms_arr_num = np.arange(len(acronyms_arr))
acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
color_map = {0: 'purple', 1: 'green', 2: 'blue', 3: 'red', 4: 'purple'}

# sessions_list
sessions_list = []

if args.data == "Allen":
    pickle_path = f"{args.data_type}/{args.data}"
    sessions_list = ['719161530', '794812542', '778998620', '798911424', '771990200', '771160300', '768515987']
elif args.data == "ibl":
    sessions_list = ['1a507308-c63a-4e02-8f32-3239a07dc578']
    pickle_path = f'{args.data_type}/{args.data}/'
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
elif args.data == "Neuronexus":
    sessions_list = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1","AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    pickle_path = f'../data/{args.data}/{args.data_type}'
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Cortex'}
print(f"Sessions List: {sessions_list}")

# def plot_embedding(data, labels, file_path):
#     ### use different markers for different sessions
    
#     # Function to plot 3D scatter plot using Plotly
#     trace = go.Scatter3d(
#         x=data[:, 0],
#         y=data[:, 1],
#         z=data[:, 2],
#         mode='markers',
#         marker=dict(
#             size=5,
#             color=[color_map[label] for label in labels],  # Color points based on labels, use 'gray' if label not found
#             opacity=0.8
#         )
#     )
#     layout = go.Layout(
#         margin=dict(l=0, r=0, b=0, t=0),
#         scene=dict(
#             xaxis_title='Component 1',
#             yaxis_title='Component 2',
#             zaxis_title='Component 3'
#         )
#     )
#     ## add color legend for points

#     fig = go.Figure(data=[trace], layout=layout)
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     fig.write_html(file_path)
def plot_embedding(data, labels, file_path, session_labels):
    # Define different markers for different sessions
    marker_symbols = ['circle', 'circle-open', 'cross', 'diamond',
            'diamond-open', 'square', 'square-open', 'x']
    session_marker_map = {session: marker_symbols[i % len(marker_symbols)] for i, session in enumerate(set(session_labels))}
    
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
    file_path = f"{output_path}/{session}_PCA.html"
    # plot_embedding(pca_data, labels, file_path, session_labels)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=pca_data[:, 0], y=pca_data[:, 1], z=pca_data[:, 2],
                                 mode='markers',
                                 marker=dict(size=5, color=[color_map[label] for label in labels], opacity=0.8),
                                 text=labels,
                                 hoverinfo='text'))
    fig.write_html(file_path)

    # 3D TSNE
    # tsne = TSNE(n_components=3)
    # tsne_data = tsne.fit_transform(data)
    # file_path = f"{output_path}/{session}_TSNE.html"
    # plot_embedding(tsne_data, labels, file_path)

    # # 3D UMAP
    # umap = UMAP(n_components=3)
    # umap_data = umap.fit_transform(data)
    # file_path = f"{output_path}/{session}_UMAP.html"
    # plot_embedding(umap_data, labels, file_path)


def plot_spectrogram(data, path):
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(data, shading='gouraud', cmap='inferno')
    plt.colorbar(label='Power (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def create_spectrogram(X, y, trial_idx, session):
    new_x = []
    for i in range(len(X)):
        # Provide axis = 0 for freq normalization and axis = 1 for time normalization
        new_x.append(compute_spectrogram_librosa(X[i], 0, axis=1))

    # Sample one random item from the list and plot it
    # random_idx = np.random.randint(len(new_x))
    # plot_spectrogram(np.transpose(new_x[random_idx]), f"{output_path}/spectrogram_example_{session}.png")

    return new_x, y, trial_idx


def create_raw(X, y, trial_idx):
    if args.data == "Allen":
        sampling_rate = 1250
    for i in range(len(X)):
        base_rms_mini = calculate_rms(X[i])
        pow_whole_mini, rms_whole_mini = calculate_power(X[i], sampling_rate)
        pow_delta_mini, rms_delta_mini = calculate_power(X[i], sampling_rate, (0, 4))
        pow_theta_mini, rms_theta_mini = calculate_power(X[i], sampling_rate, (4, 8))
        pow_alpha_mini, rms_alpha_mini = calculate_power(X[i], sampling_rate, (8, 12))
        pow_beta_mini, rms_beta_mini = calculate_power(X[i], sampling_rate, (12, 30))
        pow_gamma_mini, rms_gamma_mini = calculate_power(X[i], sampling_rate, (30, -1))

        X[i] = [base_rms_mini, pow_whole_mini, pow_delta_mini, pow_theta_mini, pow_alpha_mini,
                pow_beta_mini, pow_gamma_mini, rms_whole_mini, rms_delta_mini, rms_theta_mini,
                rms_alpha_mini, rms_beta_mini, rms_gamma_mini]

    return X, y, trial_idx


def main():
    # Visualize embedding
    for session in sessions_list:
        data = pickle.load(open(f"{pickle_path}/{session}_raw.pickle", 'rb'))
        X, y, trial_idx = zip(*[(d[0], d[1], d[2]) for d in data])
        if args.data_type == "spectrogram":
            X, y, trial_idx = create_spectrogram(X, y, trial_idx, session)
        elif args.data_type == "raw":
            X, y, trial_idx = create_raw(X, y, trial_idx)
        X = np.array(X)
        y = np.array(y)
        
        trial_idx = np.array(trial_idx)
        print(np.unique(y))
        # print nan columns
        nan_cols = np.isnan(X).any(axis=0)
        # remove nan columns
        X = X[:, ~nan_cols]
        visualize_embedding(X, y, session, {session})
    #     all_data.append(X)
    #     all_labels.append(y)
    #     all_sessions.append([i] * len(y))
    # all_data = np.concatenate(all_data, axis=0)
    # all_labels = np.concatenate(all_labels, axis=0)
    # all_sessions = np.concatenate(all_sessions, axis=0)
    # print(all_data.shape, all_labels.shape)
    # visualize_embedding(all_data, all_labels, "all_sessions", all_sessions)

    # Visualize embedding
    # for session in sessions_list:
    #     data = pickle.load(open(f"{pickle_path}/{session}_data.pickle", 'rb'))
    #     if args.data_type == "spectrogram":
    #         X, y, trial_idx = zip(*[(np.transpose(d[0]).flatten(), d[1], d[2]) for d in data])
    #     elif args.data_type == "raw":
    #         X, y, trial_idx = zip(*[(d[0], d[1], d[2]) for d in data])
    #     X = np.array(X)
    #     y = np.array(y)
    #     trial_idx = np.array(trial_idx)
    #     print(np.unique(y))
    #     # print nan columns
    #     nan_cols = np.isnan(X).any(axis=0)
    #     # remove nan columns
    #     X = X[:, ~nan_cols]

    #     visualize_embedding(X, y, session)
    ### plot all sessions together
    # load data of all sessions and concatenate
if __name__ == "__main__":
    main()