import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import f1_score


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, inputs, device):
        self.eval()
        with torch.no_grad():
            inputs = inputs.to(device).float()
            outputs = self.forward(inputs)
            _, predicted_labels = torch.max(outputs, 1)
        return predicted_labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device).float(), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
    return running_loss / len(dataloader), accuracy, f1

def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device).float()
            hidden = model.model[0](inputs)
            activated = model.model[1](hidden)
            embeddings.append(activated.cpu().numpy())
            labels.append(label.cpu().numpy())

    return np.concatenate(embeddings), np.concatenate(labels)
