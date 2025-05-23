import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, k, tau):
        """
        Adopted from deep GRAph Contrastive rEpresentation learning (GRACE): https://github.com/CRIPAC-DIG/GRACE 
        """
        super(GraphEncoder, self).__init__()
        self.k = k
        self.tau = tau
        self.activation = nn.ReLU()
        self.new_activation = nn.GELU()
        
        layers = []
        layers.append(GCNConv(in_channels, out_channels*2))

        for _ in range(1, k-1):
            layers.append(GCNConv(out_channels*2, out_channels*2))
        
        layers.append(GCNConv(out_channels*2, out_channels))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        for i in range(self.k):
            x = self.activation(self.encoder[i](x, edge_index))
        # x = self.new_activation(self.encoder[-1](x, edge_index))
        return x
    
    def cosine_sim(self, z1, z2):
        """
        Calculate the Cosine Similarity between two given views z1 and z2 of shape (N, D)
        """
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        """
        Calculate the 'semi loss' between two given views L(z1, z2) in R^n
        L(z1, z2) = l(ui, vi) for each node i 
        l(ui, vi) = positive pair / (positive pair + inter-view negative pair + intra-view negative pair)
        """
        f = lambda x: torch.exp(x / self.tau)
        intraview_pairs = f(self.cosine_sim(z1, z1))
        interview_pairs = f(self.cosine_sim(z1, z2))

        return -torch.log(
            interview_pairs.diag()
            / (intraview_pairs.sum(1) + interview_pairs.sum(1) - intraview_pairs.diag()))

    def batched_semi_loss(self, z1, z2, batch_size):
        """
        Calculate the `semi loss` between a batch of two given views
        Space complexity: O(BN) (semi_loss: O(N^2))
        """
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            # Mask out other values not in the current batch
            mask = indices[i * batch_size:(i + 1) * batch_size]

            # Similar to self.semi_loss()
            intraview_pairs = f(self.cosine_sim(z1[mask], z1))  # [B, N]
            interview_pairs = f(self.cosine_sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                interview_pairs[:, i * batch_size:(i + 1) * batch_size].diag()
                / (intraview_pairs.sum(1) + interview_pairs.sum(1)
                   - intraview_pairs[:, i * batch_size:(i + 1) * batch_size].diag())))
            # intraview_pairs = f(self.cosine_sim(z1[mask], z1[mask]))  # [B, N]
            # interview_pairs = f(self.cosine_sim(z1[mask], z2[mask]))  # [B, N]

            # losses.append(-torch.log(
            #     interview_pairs.diag()
            #     / (intraview_pairs.sum(1) + interview_pairs.sum(1)
            #        - intraview_pairs.diag())))

        return torch.cat(losses)

    def loss(self, z1, z2, mean=True, batch_size=0):
        """
        Overall objective for all positive pairs
        J = 1/(2n) * sum (l(ui, vi) + l(vi, ui))
        """
        if batch_size == 0:
            l1 = self.semi_loss(z1, z2)
            l2 = self.semi_loss(z2, z1)
        else:
            l1 = self.batched_semi_loss(z1, z2, batch_size)
            l2 = self.batched_semi_loss(z2, z1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    

class CombinedEncoder(nn.Module):
    def __init__(self, encoder_x, encoder_a, input_dim, hidden_dim, output_dim, k):
        super(CombinedEncoder, self).__init__()
        
        self.encoder_x = encoder_x
        self.encoder_a = encoder_a
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(1, k-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Sigmoid())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.ReLU())

        self.encoder_combined = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        z1 = self.encoder_a(x, edge_index)
        z2 = self.encoder_x(x)
        z = torch.cat([z1, z2], dim=1)

        output = self.encoder_combined(z)
        return z, output