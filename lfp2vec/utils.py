from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)


def compute_mask_inputs(
    model: Wav2Vec2ForPreTraining,
    input_values: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute time-mask indices and negative sample indices for SSL pretraining.

    Parameters:
    - model: wav2vec2 pretraining model providing config and feature extractor length.
    - input_values: waveform batch tensor of shape [B, T_raw].
    - device: device to place the computed indices on.

    Returns:
    - mask_time_indices: boolean mask indices over time steps [B, T_feat].
    - sampled_negatives: indices for negative sampling [B, T_feat, num_negatives].
    """
    batch_size, raw_seq_len = input_values.shape
    with torch.no_grad():
        # Compute the feature extractor output length
        seq_len = model._get_feat_extract_output_lengths(raw_seq_len).item()
        # Compute masking
        mask_time_indices = _compute_mask_indices(
            (batch_size, seq_len),
            mask_prob=model.config.mask_time_prob,
            mask_length=model.config.mask_time_length,
        )

        assert mask_time_indices.sum() > 0, "Mask time indices sum is 0"
        sampled_negatives = _sample_negative_indices(
            (batch_size, seq_len),
            num_negatives=model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        mask_time_indices = torch.tensor(mask_time_indices).to(device)
        sampled_negatives = torch.tensor(sampled_negatives).to(device)
    return mask_time_indices, sampled_negatives


def get_grad_norm(model: Wav2Vec2ForPreTraining, norm_type: float = 2.0) -> float:
    """Compute the global gradient norm across all parameters with gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


def projector_hook(module: nn.Module, input: Any, output: Any) -> None:
    """Forward hook that stores projector output for debugging/visualization."""
    global projector_output
    projector_output = output


def classifier_hook(module: nn.Module, input: Any, output: Any) -> None:
    """Forward hook that stores classifier inputs/outputs for analysis."""
    global classifier_input_first, classifier_input_whole, classifier_output
    classifier_input_first = input[0]
    classifier_input_whole = input
    classifier_output = output


def quantizer_hook(module: nn.Module, input: Any, output: Any) -> None:
    """Forward hook that logs quantizer logits and softmax probs for debugging."""
    global debug_data
    hidden_states = input[0]
    hidden_states = input[0]  # [B, T, D]
    batch_size, seq_len, hidden_size = hidden_states.shape

    proj = module.weight_proj(hidden_states)  # shape [B, T, groups * num_vars]
    logits = proj.view(
        batch_size * seq_len * module.num_groups, -1
    )  # shape [B*T*G, num_vars]

    soft_probs = torch.softmax(
        logits.view(batch_size * seq_len, module.num_groups, -1).float(), dim=-1
    )

    debug_data["logits"] = logits.detach().cpu()
    debug_data["soft_probs"] = soft_probs.detach().cpu()


def calculate_chance_accuracy(y_labels: List[str]) -> Tuple[Dict[str, int], float]:
    """Compute label counts and chance accuracy (majority class proportion)."""
    label_counts = {}
    for label in y_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    chance_accuracy = max(label_counts.values()) / sum(label_counts.values())
    return label_counts, chance_accuracy


def collect_classifier_input_embeddings(
    classification_model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Collect embeddings fed into the classifier head via a temporary forward hook.

    This runs a single no-grad forward pass over a DataLoader and captures the
    input tensor to `classification_model.classifier` for each batch.

    Returns (embeddings, labels), where labels may be None if the loader yields
    inputs only.
    """
    classification_model.eval()
    batch_embeds: List[torch.Tensor] = []
    labels_parts: List[torch.Tensor] = []

    def _hook(module: nn.Module, inputs: Any, output: Any) -> None:
        inp = inputs[0]
        batch_embeds.append(inp.detach().cpu())

    handle = classification_model.classifier.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    input_values, labels = batch
                    labels_parts.append(labels.detach().cpu())
                else:
                    input_values = (
                        batch[0] if isinstance(batch, (list, tuple)) else batch
                    )
                input_values = input_values.to(device).float()
                _ = classification_model(input_values=input_values)
        embeddings = torch.cat(batch_embeds, dim=0)
        labels_np = (
            torch.cat(labels_parts, dim=0).numpy() if len(labels_parts) > 0 else None
        )
        return embeddings.numpy(), labels_np
    finally:
        handle.remove()
