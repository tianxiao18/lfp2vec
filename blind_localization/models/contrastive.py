import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import vit_b_16
from transformers import ViTImageProcessor
from transformers import ASTForAudioClassification

class RawSignalDataset(Dataset):
    def __init__(self, signals, labels, transform=None):
        self.signals = signals
        self.labels = labels
        self.transform = transform
        self.label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        # Positive sample: randomly choose within the same label
        if len(self.label_to_indices[label]) > 1:
            positive_idx = idx
            while positive_idx == idx:
                positive_idx = np.random.choice(self.label_to_indices[label])
        else:
            positive_idx = np.random.choice(len(self.labels))

        # Negative sample: randomly choose from a different label
        negative_label = np.random.choice(list(self.label_to_indices.keys() - {label}))
        negative_idx = np.random.choice(self.label_to_indices[negative_label])

        pos_signal = self.signals[positive_idx]
        neg_signal = self.signals[negative_idx]

        if self.transform:
            signal = self.transform(signal)
            pos_signal = self.transform(pos_signal)
            neg_signal = self.transform(neg_signal)

        signal = torch.tensor(signal, dtype=torch.float32)
        pos_signal = torch.tensor(pos_signal, dtype=torch.float32)
        neg_signal = torch.tensor(neg_signal, dtype=torch.float32)

        return (signal, pos_signal, neg_signal), label


class ContrastiveEncoder(nn.Module):
    def __init__(self, fc_layer_size, input_size, output_size):
        super(ContrastiveEncoder, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, fc_layer_size),
            nn.ReLU(),
            nn.Linear(fc_layer_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)

        # compute similarity for all pairs of z_i, z_j
        sim = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        mask = torch.ones_like(sim).fill_diagonal_(0)
        sim = sim * mask

        # exponentiate positive pair similarities between z_i, z_j
        pos_sim_ij = torch.diag(sim, z_i.size(0))
        pos_sim_ji = torch.diag(sim, -z_i.size(0))
        exp_pos_sim = torch.exp(torch.cat([pos_sim_ij, pos_sim_ji], dim=0))

        # sum exponentials of non-self positive similarities
        sum_exp_reg = torch.sum(torch.exp(sim), dim=1, keepdim=True)

        loss = -torch.log(exp_pos_sim / sum_exp_reg)
        return loss.mean()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5, device='cpu'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

    def forward(self, z_i, z_j):
        """
        Adopted from SimCLR loss: https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        features = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([torch.arange(len(z_i)) for i in range(2)])
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        if torch.isnan(positives).any() or torch.isnan(negatives).any():
            print("NaN detected in positives or negatives")
        if torch.isinf(positives).any() or torch.isinf(negatives).any():
            print("Inf detected in positives or negatives")

        logits = torch.cat([positives, negatives], dim=1)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN or Inf detected in logits")
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        loss = self.criterion(logits, labels)
        return loss

class ResNetEncoder(nn.Module):
    def __init__(self, input_size, output_size, use_projector=False, pretrained=True):
        super(ResNetEncoder, self).__init__()

        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_features = self.resnet.fc.in_features
        self.use_projector = use_projector
        self.resnet.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        self.feature_adjust = nn.Linear(num_features, output_size)

    def forward(self, x):
        x = self.resnet(x)
        if self.use_projector:
            x = self.projector(x)
        else:
            x = self.feature_adjust(x)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, input_size, output_size, use_projector=False, pretrained=True, freq_bins=500):
        super(ViTEncoder, self).__init__()

        self.vit = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", attn_implementation="sdpa", torch_dtype=torch.float16)
        self.vit.classifier = nn.Identity()
        
        self.use_projector = use_projector
        hidden_dim = self.vit.audio_spectrogram_transformer.layernorm.normalized_shape[0]

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        ).half()

        self.feature_adjust = nn.Linear(hidden_dim, output_size).half()
        self.freq_bins = freq_bins

    def forward(self, x):        
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True)) / 2
        x = x.half()
        x = self.vit(x).logits

        if self.use_projector:
            x = self.projector(x)
        else:
            x = self.feature_adjust(x)

        return x
    
