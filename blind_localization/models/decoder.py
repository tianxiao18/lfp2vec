import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np
import pickle
from sklearn.metrics import f1_score, balanced_accuracy_score


class ContrastiveLearningWithMLP(nn.Module):
    def __init__(self, encoder, input_dim, hidden_dim, output_dim):
        super(ContrastiveLearningWithMLP, self).__init__()

        self.encoder = encoder if encoder is not None else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        output = self.mlp(z)
        return z, output


class ContrastiveLearningWithLR(nn.Module):
    def __init__(self, encoder, input_dim, output_dim):
        super(ContrastiveLearningWithLR, self).__init__()

        self.encoder = encoder if encoder is not None else nn.Identity()
        self.lr = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        output = self.lr(z)
        return z, output


def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def train_decoder(model, dataloader, optimizer, contrastive_criterion, supervised_criterion, device='cpu',
                  mode="separate", lambda_c=1):
    """
    Train model in different modes, here model is encoder+decoder combined
    - "separate": freeze the pretrained encoder + train the decoder separately
    - "end_to_end": fine tune both encoder and decoder jointly
    """
    if mode == "separate":
        freeze_model(model.encoder)
    else:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True

    model.train()
    total_loss = []
    total_samples, correct_predictions = 0, 0
    all_true_labels = []
    all_predicted_labels = []

    for data, label in dataloader:
        spectrograms, augmented_spectrograms = data
        spectrograms = spectrograms.to(device)
        augmented_spectrograms = augmented_spectrograms.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        z_i, output_i = model(spectrograms)
        z_j, _ = model(augmented_spectrograms)

        if mode == "separate":
            contrastive_loss = 0
        else:
            contrastive_loss = contrastive_criterion(z_i, z_j)

        label = label.long()
        supervised_loss = supervised_criterion(output_i, label)
        loss = contrastive_loss + lambda_c * supervised_loss

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        _, predicted = torch.max(output_i, 1)
        all_true_labels.extend(label.cpu().numpy())
        all_predicted_labels.extend(predicted.cpu().numpy())
        correct_predictions += (predicted == label).sum().item()
        total_samples += label.size(0)

    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = correct_predictions / total_samples
    balanced_accuracy = balanced_accuracy_score(all_true_labels, all_predicted_labels)
    metrics = {"loss": avg_loss, "balanced_accuracy": balanced_accuracy, "accuracy": accuracy,
                "real_labels": all_true_labels,"predicted_labels": all_predicted_labels}

    return metrics


def validate_decoder(model, dataloader, supervised_criterion, device='cpu'):
    model.eval()
    total_loss = []
    total_samples, correct_predictions = 0, 0
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_logits = []
    all_embeddings = []

    with torch.no_grad():
        for data, label in dataloader:
            spectrograms, _ = data
            spectrograms = spectrograms.to(device)
            label = label.to(device)
            label = label.long()

            z, output = model(spectrograms)
            loss = supervised_criterion(output, label)

            total_loss.append(loss.item())
            _, predicted = torch.max(output, 1)
            all_true_labels.extend(label.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            all_predicted_logits.extend(output.cpu().numpy())
            all_embeddings.extend(z.cpu().numpy())

            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = correct_predictions / total_samples

    metrics = {"loss": avg_loss,
               "accuracy": accuracy,
               "balanced_accuracy": balanced_accuracy_score(np.array(all_true_labels), np.array(all_predicted_labels)),
               "f1": f1_score(np.array(all_true_labels), np.array(all_predicted_labels), average='micro'),
               "weighted_f1": f1_score(np.array(all_true_labels), np.array(all_predicted_labels), average='weighted'),
               "macro_f1": f1_score(np.array(all_true_labels), np.array(all_predicted_labels), average='macro'),
               "real_labels": all_true_labels,
               "predicted_labels": all_predicted_labels,
               "logits":all_predicted_logits,
               "embeddings":all_embeddings}
    return metrics


class GraphDecoder(nn.Module):
    def __init__(self, encoder, input_dim, hidden_dim, output_dim):
        super(GraphDecoder, self).__init__()

        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        output = self.mlp(z)
        return z, output
