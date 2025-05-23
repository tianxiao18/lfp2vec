import math
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class BaseCriterion(nn.Module):
    def __init__(self):
        super(BaseCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        raise NotImplementedError

    def forward(self, model, batch, device):
        raise NotImplementedError
    

class PretrainMaskedCriterion(BaseCriterion):
    def __init__(self, alpha):
        super(PretrainMaskedCriterion, self).__init__()
        self.alpha = alpha

    def forward(self, output: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor):
        predicted = output.masked_select(mask)
        true_activity = truth.masked_select(mask)
        l1 = torch.mean(torch.abs(true_activity - predicted))
        non_zero_idxs = torch.abs(true_activity) > 1
        non_zero = torch.mean(torch.abs(true_activity[non_zero_idxs] - predicted[non_zero_idxs]))
        content_aware_loss = self.alpha * non_zero
        loss = l1 + content_aware_loss
        return loss


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def build_model(self, cfg):
        raise NotImplementedError

    def save_model_weights(self, states):
        #expects a new state with "models" key
        states["model"] = self.state_dict() 
        return states

    def load_weights(self, states):
        self.load_state_dict(states)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        '''
        From https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308/2
        '''
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, seq):
        #seq is [batch, len, dim]
        assert len(seq.shape) == 3
        pos_enc = self.pe[:,:seq.size(1),:]
        out = seq + pos_enc
        test = torch.zeros_like(seq) + pos_enc
        return out, pos_enc


class TransformerEncoderInput(nn.Module):
    def __init__(self, cfg, dropout=0.1):
        super(TransformerEncoderInput, self).__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(in_features=cfg.input_dim, out_features=cfg.hidden_dim)
        self.positional_encoding = PositionalEncoding(self.cfg.hidden_dim)
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_specs):
        input_specs = self.in_proj(input_specs)
        input_specs, pos_enc = self.positional_encoding(input_specs)
        input_specs = self.layer_norm(input_specs)
        input_specs = self.dropout(input_specs)
        return input_specs, pos_enc


class SpecPredictionHead(nn.Module): 
    def __init__(self, cfg):
        super(SpecPredictionHead, self).__init__()
        self.hidden_layer = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.act_fn = None
        if cfg.layer_activation=="gelu":
            self.act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output = nn.Linear(cfg.hidden_dim, cfg.input_dim)

    def forward(self, hidden):
        h = self.hidden_layer(hidden)
        h = self.act_fn(h)
        h = self.layer_norm(h)
        h = self.output(h)
        return h


class BrainBERTModel(BaseModel):
    def __init__(self):
        super(BrainBERTModel, self).__init__()

    def forward(self, input_specs: torch.Tensor, src_key_mask=None, rep_from_layer=-1):

        input_specs, pos_enc = self.input_encoding(input_specs)
        input_specs = input_specs.transpose(0,1) #nn.Transformer wants [seq, batch, dim]
        if rep_from_layer==-1:
            output_specs: torch.Tensor = self.transformer(input_specs, src_key_padding_mask=src_key_mask)
        else:
            raise NotImplementedError
        output_specs = output_specs.transpose(0,1) #[batch, seq, dim]
        intermediate_rep = output_specs
        output_specs = self.spec_prediction_head(output_specs)
        return output_specs, pos_enc, intermediate_rep

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.bias.data.fill_(1.0)

    def build_model(self, cfg):
        self.cfg = cfg
        hidden_dim = self.cfg.hidden_dim
        self.input_encoding = TransformerEncoderInput(cfg)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=self.cfg.nhead,
                                                   dim_feedforward=self.cfg.layer_dim_feedforward,
                                                   activation=self.cfg.layer_activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.encoder_num_layers)
        self.spec_prediction_head = SpecPredictionHead(cfg)
        self.apply(self.init_weights)


def train(model, dataloader: DataLoader, optimizer: Optimizer, criterion, device='cpu'):
    
    model.train()
    total_loss = []

    for (spectrograms, augmented_spectrograms), _ in dataloader:
        input: torch.Tensor = augmented_spectrograms
        truth: torch.Tensor = spectrograms

        input = input.reshape((input.shape[0], 500, -1)).transpose(1, 2)  # (input.shape[0], 4, 500)
        mask = torch.all(input==0, dim=2).unsqueeze(2)                    # (input.shape[0], 4,   1)
        truth = truth.reshape((input.shape[0], 500, -1)).transpose(1, 2)  # (input.shape[0], 4, 500)

        # assert input.shape == (input.shape[0], 4, 500), input.shape
        # assert mask.shape  == (input.shape[0], 4,   1), mask.shape
        # assert truth.shape == (input.shape[0], 4, 500), truth.shape

        input = input.to(device)
        mask  = mask.to(device)
        truth = truth.to(device)
        optimizer.zero_grad()

        pred = model(input)[0]
        loss: torch.Tensor = criterion(pred, truth, mask)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss


def validation(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_loss = []

    with torch.no_grad():
        for (spectrograms, augmented_spectrograms), _ in dataloader:
            input: torch.Tensor = augmented_spectrograms
            truth: torch.Tensor = spectrograms

            input = input.reshape((input.shape[0], 500, -1)).transpose(1, 2)  # (input.shape[0], 4, 500)
            mask = torch.all(input==0, dim=2).unsqueeze(2)                    # (input.shape[0], 4,   1)
            truth = truth.reshape((input.shape[0], 500, -1)).transpose(1, 2)  # (input.shape[0], 4, 500)

            # assert input.shape == (input.shape[0], 4, 500), input.shape
            # assert mask.shape == (input.shape[0], 4, 1), mask.shape
            # assert truth.shape == (input.shape[0], 4, 500), truth.shape

            input = input.to(device)
            mask  = mask.to(device)
            truth = truth.to(device)

            pred = model(input)[0]
            loss: torch.Tensor = criterion(pred, truth, mask)
            total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss


class BrainBERTWithMLP(nn.Module):
    def __init__(self, encoder: BrainBERTModel, input_dim, hidden_dim, output_dim):
        super(BrainBERTWithMLP, self).__init__()

        self.encoder: BrainBERTModel = encoder
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        recon_input, _, inter_rep = self.encoder(x)
        output = self.mlp(inter_rep)
        return recon_input, output
    

def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


def train_decoder(model: BrainBERTWithMLP, dataloader, optimizer: Optimizer,
                  encoder_criterion, supervised_criterion, device='cpu',
                  mode="separate"):
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

    for (spectrograms, augmented_spectrograms), label in dataloader:

        augmented_spectrograms: torch.Tensor = augmented_spectrograms
        spectrograms          : torch.Tensor = spectrograms
        label                 : torch.Tensor = label

        augmented_spectrograms = augmented_spectrograms.reshape((augmented_spectrograms.shape[0], 500, -1)).transpose(1, 2)  # (augmented_spectrograms.shape[0], 4, 500)
        mask = torch.all(augmented_spectrograms==0, dim=2).unsqueeze(2)                                                      # (augmented_spectrograms.shape[0], 4,   1)
        spectrograms = spectrograms.reshape((augmented_spectrograms.shape[0], 500, -1)).transpose(1, 2)                      # (augmented_spectrograms.shape[0], 4, 500)
        
        # assert augmented_spectrograms.shape == (augmented_spectrograms.shape[0], 4, 500), augmented_spectrograms.shape
        # assert mask.shape                   == (augmented_spectrograms.shape[0], 4,   1), mask.shape
        # assert spectrograms.shape           == (augmented_spectrograms.shape[0], 4, 500), spectrograms.shape

        augmented_spectrograms = augmented_spectrograms.to(device)
        mask = mask.to(device)
        spectrograms = spectrograms.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        recon_input_i, output_i = model(augmented_spectrograms)

        if mode == "separate":
            encoder_loss = 0
        else:
            encoder_loss = encoder_criterion(recon_input_i, spectrograms, mask)

        label = label.long()
        supervised_loss = supervised_criterion(output_i, label)
        loss: torch.Tensor = encoder_loss + supervised_loss

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        _, predicted = torch.max(output_i, 1)
        correct_predictions += (predicted == label).sum().item()
        total_samples += label.size(0)

    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


def validate_decoder(model: BrainBERTWithMLP, dataloader,
                     supervised_criterion, device='cpu'):
    model.eval()
    total_loss = []
    total_samples, correct_predictions = 0, 0
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for (spectrograms, _), label in dataloader:

            spectrograms: torch.Tensor = spectrograms
            label       : torch.Tensor = label

            spectrograms = spectrograms.reshape((spectrograms.shape[0], 500, -1)).transpose(1, 2)  # (augmented_spectrograms.shape[0], 4, 500)
            # assert spectrograms.shape == (spectrograms.shape[0], 4, 500), spectrograms.shape

            spectrograms = spectrograms.to(device)
            label = label.to(device)
            label = label.long()

            _, output = model(spectrograms)
            loss: torch.Tensor = supervised_criterion(output, label)
            total_loss.append(loss.item())

            _, predicted = torch.max(output, 1)
            all_true_labels.extend(label.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = correct_predictions / total_samples
    f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
    return avg_loss, accuracy, f1
