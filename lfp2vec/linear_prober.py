import torch
import torch.nn as nn


class LinearProber(nn.Module):
    def __init__(self, encoder: nn.Module, rep_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(rep_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            reps = self.encoder(x).last_hidden_state.detach()  # shape [B, T, D]
            reps = reps.mean(dim=1)  # first token pooling, also consider mean pooling
        return self.classifier(reps)
