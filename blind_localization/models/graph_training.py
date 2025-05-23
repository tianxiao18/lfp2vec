import torch.nn as nn
from torch_geometric.utils import dropout_edge
from blind_localization.models.decoder import freeze_model
from blind_localization.visualize import visualize_explained_variance, visualize_graph
import torch
import pickle
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score 
import networkx as nx
import matplotlib.pyplot as plt

def drop_feature(spectrograms, percent_masked=0.2, mask_size=1):
    """do temporal masking for a list of spectrograms """
    n_channels, n_features = spectrograms.size()
    n_masks = int(n_features * percent_masked)
    start_indices = torch.randint(0, n_features-mask_size+1, (n_channels, n_masks))

    offsets = torch.arange(mask_size)
    mask_indices = start_indices.unsqueeze(2) + offsets
    mask_indices = mask_indices.view(n_channels, -1)

    mask = torch.ones((n_channels, n_features), dtype=torch.bool)
    batch_indices = torch.arange(n_channels, device=spectrograms.device).unsqueeze(1).expand_as(mask_indices)

    batch_indices_flat = batch_indices.reshape(-1)
    mask_indices_flat = mask_indices.reshape(-1)
    mask[batch_indices_flat[mask_indices_flat < n_features], mask_indices_flat[mask_indices_flat < n_features]] = False

    masked_spectrograms = spectrograms * mask.float()
    # n_channels, n_freq_bins, n_temp_bins = spectrograms.size()
    # n_masks = int(n_temp_bins * percent_masked)

    # for i in range(len(spectrograms)):
    #     for _ in range(n_masks):
    #         start_idx = torch.randint(0, spectrograms[0].size(0) - mask_size, (1,)).item()
    #         spectrograms[i][:, start_idx:start_idx+mask_size] = 0

    # n_masks = int(n_freq_bins * percent_masked)
    # for i in range(len(spectrograms)):
    #     for _ in range(n_masks):
    #         start_idx = torch.randint(0, spectrograms[0].size(0) - mask_size, (1,)).item()
    #         spectrograms[i][start_idx:start_idx+mask_size, :] = 0
    
    # spectrograms = spectrograms.reshape(n_channels, n_freq_bins*n_temp_bins)
    
    return masked_spectrograms

def visualize_graph_with_specific_colors(x, edge_index, labels, title="augmented_graph1"):
    edge_index_np = edge_index.cpu().numpy()
    G = nx.Graph()

    for edge in edge_index_np.T:
        G.add_edge(edge[0], edge[1])

    plt.figure(figsize=(8, 8))
    specific_colors = {0: 'red', 1: 'orange', 2: 'green', 3: 'blue', 4: 'purple'}
    node_colors = [specific_colors[label] for label in labels.cpu().numpy()]
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300)
    
    plt.savefig(f'{title}.png')

def train_graph_encoder(model, dataloader, optimizer, device, session_config, visualize=False):
    model.train()
    total_loss = []

    for data in dataloader:
        # Here each data point in dataloader represents one graph G = (X, A)  
        # Masking Node Features (MF)
        x_1 = drop_feature(data.x, session_config['percent_masked'], session_config['mask_size']).to(device)
        x_2 = drop_feature(data.x, session_config['percent_masked'], session_config['mask_size']).to(device)
        
        edge_index_1 = dropout_edge(data.edge_index, p=session_config['drop_edge_rate'])[0].to(device)
        edge_index_2 = dropout_edge(data.edge_index, p=session_config['drop_edge_rate'])[0].to(device)

        if visualize:
            data.edge_index = edge_index_1
            visualize_graph(data, device, title="augmented_graph1")
            data.edge_index = edge_index_2
            visualize_graph(data, device, title="augmented_graph2")
            visualize=False

        # Generating views
        z_i = model(x_1, edge_index_1)
        z_j = model(x_2, edge_index_2)

        # loss is computed between all pair of nodes in two graphs
        loss = model.loss(z_i, z_j, batch_size=data.batch.max().item() + 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss

def validate_graph_encoder(model, dataloader, device, session_config):
    model.eval()
    total_loss = []

    with torch.no_grad():
        for data in dataloader:
            x_1 = drop_feature(data.x, session_config['percent_masked'], session_config['mask_size']).to(device)
            x_2 = drop_feature(data.x, session_config['percent_masked'], session_config['mask_size']).to(device)
            
            edge_index_1 = dropout_edge(data.edge_index, p=session_config['drop_edge_rate'])[0].to(device)
            edge_index_2 = dropout_edge(data.edge_index, p=session_config['drop_edge_rate'])[0].to(device)

            z_i = model(x_1, edge_index_1)
            z_j = model(x_2, edge_index_2)

            loss = model.loss(z_i, z_j, batch_size=data.batch.max().item() + 1)
            total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss

def train_graph_decoder(model, dataloader, optimizer, supervised_criterion, session_config, device='cpu', separate=True,
                        lambda_c=1):
    """
    Train model in different modes, here model is encoder+decoder combined
    - "separate": freeze the pretrained encoder + train the decoder separately
    - "end_to_end": fine tune both encoder and decoder jointly
    """
    if separate:
        freeze_model(model.encoder)
    else:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True

    model.train()
    total_loss = []
    all_embeddings = []
    total_samples, correct_predictions = 0, 0

    for data in dataloader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        label = data.y.to(device)

        z, output = model(x, edge_index)
        supervised_loss = supervised_criterion(output, label)
        
        if separate:
            contrastive_loss = 0
        else:
            x_1 = drop_feature(data.x, session_config['percent_masked'], session_config['mask_size']).to(device)
            x_2 = drop_feature(data.x, session_config['percent_masked'], session_config['mask_size']).to(device)
            
            edge_index_1 = dropout_edge(data.edge_index, p=session_config['drop_edge_rate'])[0].to(device)
            edge_index_2 = dropout_edge(data.edge_index, p=session_config['drop_edge_rate'])[0].to(device)

            z_i = model.encoder(x_1, edge_index_1)
            z_j = model.encoder(x_2, edge_index_2)
            contrastive_loss = model.encoder.loss(z_i, z_j, batch_size=0)

        loss = contrastive_loss + lambda_c * supervised_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == label).sum().item()
        total_samples += label.size(0)
        all_embeddings.append(z.detach().cpu().numpy())

    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = correct_predictions / total_samples
    metrics = {"loss":avg_loss, "accuracy": accuracy}
    all_embeddings = np.vstack(all_embeddings)
    if session_config['visualize']: visualize_explained_variance(all_embeddings)

    return metrics


def validate_graph_decoder(model, dataloader, supervised_criterion, device='cpu', pickle_path=None):
    model.eval()
    total_loss = []
    total_samples, correct_predictions = 0, 0
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for data in dataloader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            label = data.y.to(device)

            z, output = model(x, edge_index)
            loss = supervised_criterion(output, label)
            total_loss.append(loss.item())
            
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)
    
    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = correct_predictions / total_samples

    if pickle_path: 
         with open(pickle_path, 'wb') as f:
                pickle.dump((all_predictions, all_labels), f)

    metrics = {"loss":avg_loss, 
               "accuracy": accuracy, 
               "balanced_accuracy": balanced_accuracy_score(np.array(all_labels), np.array(all_predictions)),
                "f1":f1_score(np.array(all_labels),  np.array(all_predictions), average='micro'),
                "weighted_f1":f1_score(np.array(all_labels),  np.array(all_predictions), average='weighted'),
                "macro_f1": f1_score(np.array(all_labels),  np.array(all_predictions), average='macro')}

    return metrics
