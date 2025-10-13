import gc
import logging
import os
import pickle
import tempfile
from typing import Tuple
from uuid import uuid4
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import evaluate
import numpy as np
import torch
import torch.nn as nn
import wandb
from dataloader import LFP2VecDataLoader
from linear_prober import LinearProber
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
)
from utils import (
    calculate_chance_accuracy,
    classifier_hook,
    collect_classifier_input_embeddings,
    compute_mask_inputs,
    get_grad_norm,
    projector_hook,
    quantizer_hook,
)

from blind_localization.data.PCAviz import PCAVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training(
    data: str = "Allen",
    data_type: str = "spectrogram_preprocessed",
    val_size: float = 0.2,
    test_size: float = 0.2,
    sampling_rate: int = 1250,
    rand_init: bool = False,
    ssl: bool = True,
    epoch: int = 50,
    lr: float = 1e-5,
):
    """End-to-end pipeline: pretraining (optional), embedding viz, fine-tuning, logging.

    - Builds datasets, optionally does wav2vec2 SSL pretraining, visualizes classifier-input
      embeddings pre/post FT, fine-tunes classification head, and saves results.
    """

    # tags
    ri_tag = "rand_init" if rand_init else "pretrained"
    ssl_tag = "ssl" if ssl else "nossl"
    ft_tag = "no_ft"

    # output path
    output_path = f"../results/{data}/{data_type}/wave2vec2/across_session"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # load data
    data_loader = LFP2VecDataLoader(data, val_size, test_size)
    train_dataset, val_dataset, test_dataset = data_loader.parse_datasets(
        sampling_rate=sampling_rate
    )

    # id2label, label2id
    acronyms_arr = data_loader.hc_acronyms
    id2label = {str(i): acr for i, acr in enumerate(acronyms_arr)}
    label2id = {acr: str(i) for i, acr in enumerate(acronyms_arr)}

    logger.info(f"label2id: {label2id}")
    logger.info(f"id2label: {id2label}")
    logger.info("Generating Wav2Vec Config...")
    w2v2_config = Wav2Vec2Config(
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        final_dropout=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        feat_extract_norm="group",
        feat_proj_dropout=0.0,
        feat_extract_activation="gelu",
        feat_quantizer_dropout=0.0,
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 3, 3),
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embeddings_groups=16,
        do_stable_layer_norm=False,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        num_codevectors_per_group=320,
        num_codevector_groups=2,
        contrastive_logits_temperature=0.1,
        num_negatives=100,
        codevector_dim=256,
        proj_codevector_dim=256,
        diversity_loss_weight=0.1,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        # custom
        mask_time_min_masks=2,
        random_init=rand_init,
        self_supervised=ssl,
    )

    # convert to dict
    w2v2_config_dict = w2v2_config.to_dict()

    logger.info("Initializing WandB...")
    wandb.init(
        project="lfp2vec",
        config=w2v2_config_dict,
        name=(f"{data}-" f"{ri_tag}-" f"{ssl_tag}-exp"),
        reinit=True,
    )
    logger.info("Initializing Model...")
    if rand_init:
        ssl_model = Wav2Vec2ForPreTraining(config=w2v2_config)
    else:
        ssl_model = Wav2Vec2ForPreTraining.from_pretrained(
            "facebook/wav2vec2-base", config=w2v2_config, ignore_mismatched_sizes=True
        )

    ssl_model.quantizer.register_forward_hook(quantizer_hook)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_model.to(device)
    logger.info(f"Model is on device: {ssl_model.device}")

    logger.info("Training the model...")
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    # test_loader not needed for SSL training loop here
    max_probe_acc = 0

    if ssl:
        for epoch in tqdm(range(epoch)):
            train_loss, grad_norm = train(ssl_model, train_loader, optimizer, device)
            val_loss = validate(ssl_model, val_loader, device)
            logger.info(
                f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Grad Norm: {grad_norm:.4f}"
            )
            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Grad Norm": grad_norm,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            if (epoch + 1) % 1 == 0:
                probe_train_loss, probe_train_acc, probe_val_loss, probe_val_acc = (
                    train_probe(
                        ssl_model.wav2vec2,
                        train_loader,
                        val_loader,
                        w2v2_config.hidden_size,
                        len(id2label),
                        device,
                    )
                )
                logger.info(
                    f"Probe Train Loss: {probe_train_loss:.4f}, Probe Train Accuracy: {probe_train_acc:.4f}, "
                )
                logger.info(
                    f"Probe Val Loss: {probe_val_loss:.4f}, Probe Val Accuracy: {probe_val_acc:.4f}"
                )
                wandb.log(
                    {
                        "Probe Train Loss": probe_train_loss,
                        "Probe Train Accuracy": probe_train_acc,
                        "Probe Val Loss": probe_val_loss,
                        "Probe Val Accuracy": probe_val_acc,
                    }
                )
                if max_probe_acc > probe_val_acc:
                    max_probe_acc = max(max_probe_acc, probe_val_acc)
                    ssl_model.save_pretrained(f"{output_path}/disease/ssl_model/")

    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base",
        # num_labels=len(id2label),
        # label2id=label2id,
        # id2label=id2label,
        config=w2v2_config,
    )

    # Check if checkpoint exists
    if os.path.exists(f"{output_path}/disease/ssl_model/"):
        ssl_model = Wav2Vec2ForPreTraining.from_pretrained(
            f"{output_path}/disease/ssl_model/",
        )
        model.wav2vec2.load_state_dict(ssl_model.wav2vec2.state_dict())
    

    training_args = TrainingArguments(
        output_dir=f"{output_path}/disease",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=12,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        fp16=True,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    uuid = uuid4().hex
    unique_cache_dir = tempfile.mkdtemp(prefix="hf_eval_")
    accuracy = evaluate.load(
        "accuracy", experiment_id=str(uuid), cache_dir=unique_cache_dir
    )

    def compute_metrics(eval_pred):
        """Compute accuracy from Trainer eval predictions."""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

    # Setup trainer (used later for fine-tuning)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
    )

    # Register hooks
    projector_hook_handle = model.projector.register_forward_hook(projector_hook)
    classifier_hook_handle = model.classifier.register_forward_hook(classifier_hook)

    # Prepare for embedding collection before fine-tuning
    model.to(device)
    train_eval_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_eval_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_eval_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Efficient single-pass embedding collection via classifier input hook
    train_embeddings, train_labels = collect_classifier_input_embeddings(
        model, train_eval_loader, device
    )
    val_embeddings, val_labels = collect_classifier_input_embeddings(
        model, val_eval_loader, device
    )
    test_embeddings, test_labels = collect_classifier_input_embeddings(
        model, test_eval_loader, device
    )

    visualizer = PCAVisualizer(
        {str(i): region for i, region in enumerate(acronyms_arr)},
        output_path=output_path,
    )
    try:
        visualizer.create_pca(
            train_embeddings,
            train_labels,
            2,
            f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
            "train",
        )
        visualizer.create_pca(
            train_embeddings,
            train_labels,
            3,
            f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
            "train",
        )
        visualizer.create_pca(
            val_embeddings, val_labels, 2, f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}", "val"
        )
        visualizer.create_pca(
            val_embeddings, val_labels, 3, f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}", "val"
        )
        visualizer.create_pca(
            test_embeddings,
            test_labels,
            2,
            f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
            "test",
        )
        visualizer.create_pca(
            test_embeddings,
            test_labels,
            3,
            f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
            "test",
        )

        datasets = {
            "train": {"embeddings": train_embeddings, "labels": train_labels},
            "val": {"embeddings": val_embeddings, "labels": val_labels},
            "test": {"embeddings": test_embeddings, "labels": test_labels},
        }
        visualizer.create_combined_pca(
            datasets, 2, f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}", "combined"
        )
        visualizer.create_combined_pca(
            datasets, 3, f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}", "combined"
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(
            f"Possible size mismatch: "
            f"Train({len(train_embeddings)}, {len(train_labels)}), "
            f"Val({len(val_embeddings)}, {len(val_labels)}), "
            f"Test({len(test_embeddings)}, {len(test_labels)})"
        )

    # fine tune the model here
    best_ckpt_path = None
    trainer.train()
    best_ckpt_path = trainer.state.best_model_checkpoint

    # pickle relevant data
    train_pred = trainer.predict(train_dataset.with_format("torch"))
    train_logits = train_pred[0]
    train_labels = train_pred[1]
    train_acc = train_pred[2]["test_accuracy"]

    val_pred = trainer.predict(val_dataset.with_format("torch"))
    val_logits = val_pred[0]
    val_labels = val_pred[1]
    val_acc = val_pred[2]["test_accuracy"]

    test_pred = trainer.predict(test_dataset.with_format("torch"))
    test_logits = test_pred[0]
    test_labels = test_pred[1]
    test_acc = test_pred[2]["test_accuracy"]

    train_embeddings, train_labels = collect_classifier_input_embeddings(
        model, train_eval_loader, device
    )
    val_embeddings, val_labels = collect_classifier_input_embeddings(
        model, val_eval_loader, device
    )
    test_embeddings, test_labels = collect_classifier_input_embeddings(
        model, test_eval_loader, device
    )

    ft_tag = "ft"
    visualizer = PCAVisualizer(
        {str(i): region for i, region in enumerate(acronyms_arr)},
        output_path=output_path,
    )
    try:
        visualizer.create_pca(
            train_embeddings,
            train_labels,
            2,
            f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
            "train",
        )
        visualizer.create_pca(
            train_embeddings,
            train_labels,
            3,
            f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
            "train",
        )
        visualizer.create_pca(
            val_embeddings, val_labels, 2, f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}", "val"
        )
        visualizer.create_pca(
            val_embeddings, val_labels, 3, f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}", "val"
        )
        visualizer.create_pca(
            test_embeddings,
            test_labels,
            2,
            f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
            "test",
        )
        visualizer.create_pca(
            test_embeddings,
            test_labels,
            3,
            f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
            "test",
        )

        datasets = {
            "train": {"embeddings": train_embeddings, "labels": train_labels},
            "val": {"embeddings": val_embeddings, "labels": val_labels},
            "test": {"embeddings": test_embeddings, "labels": test_labels},
        }
        visualizer.create_combined_pca(
            datasets, 2, f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}", "combined"
        )
        visualizer.create_combined_pca(
            datasets, 3, f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}", "combined"
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(
            f"Possible size mismatch: "
            f"Train({len(train_embeddings)}, {len(train_labels)}), "
            f"Val({len(val_embeddings)}, {len(val_labels)}), "
            f"Test({len(test_embeddings)}, {len(test_labels)})"
        )

    file_path = os.path.join(
        output_path, f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}_results.pickle"
    )

    # chance accuracy
    train_label_counts, train_chance_accuracy = calculate_chance_accuracy(
        train_dataset.labels
    )
    val_label_counts, val_chance_accuracy = calculate_chance_accuracy(
        val_dataset.labels
    )
    test_label_counts, test_chance_accuracy = calculate_chance_accuracy(
        test_dataset.labels
    )

    file_obj = {
        "train_logits": train_logits,  # Predicted logits for training set
        "train_labels": train_labels,  # True labels for training set
        "train_acc": train_acc,  # Accuracy for training set
        # "train_trial_chans": train_dataset.get_trial_chan(), TBD!!!
        "train_embeddings": train_embeddings,  # Projector layer Embeddings for training set
        "val_logits": val_logits,  # Predicted logits for validation set
        "val_labels": val_labels,  # True labels for validation set
        "val_acc": val_acc,  # Accuracy for validation set
        "val_embeddings": val_embeddings,  # Projector layer Embeddings for validation set
        "test_logits": test_logits,  # Predicted logits for test set
        "test_labels": test_labels,  # True labels for test set
        "test_acc": test_acc,  # Accuracy for test set
        "test_embeddings": test_embeddings,  # Projector layer Embeddings for test set
        "train_label_counts": train_label_counts,  # Label counts for training set
        "train_chance_accuracy": train_chance_accuracy,  # Chance accuracy for training set
        "val_label_counts": val_label_counts,  # Label counts for validation set
        "val_chance_accuracy": val_chance_accuracy,  # Chance accuracy for validation set
        "test_label_counts": test_label_counts,  # Label counts for test set
        "test_chance_accuracy": test_chance_accuracy,  # Chance accuracy for test set
        "w2v2_config": w2v2_config_dict,  # Model configuration
        "best_ckpt_path": best_ckpt_path,  # Path to the best checkpoint
    }

    with open(file_path, "wb") as f:
        pickle.dump(file_obj, f)
        logger.info(f"Session {data} results saved to {file_path}")
    logger.info(
        f"Train accuracy: {train_acc}, Validation accuracy: {val_acc}, Test accuracy: {test_acc}"
    )
    wandb.log(
        {
            "Train accuracy": train_acc,
            "Validation accuracy": val_acc,
            "Test accuracy": test_acc,
        }
    )

    # Deregister hooks
    projector_hook_handle.remove()
    classifier_hook_handle.remove()
    gc.collect()
    torch.cuda.empty_cache()


def train(
    model: Wav2Vec2ForPreTraining,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """One SSL training epoch for wav2vec2 pretraining."""

    total_loss = 0
    grad_norm = []
    model.train()
    for (input_values, labels) in train_loader:
        input_values = input_values.float().to(device)
        mask_time_indices, sampled_negative_indices = compute_mask_inputs(
            model, input_values, device
        )
        sampled_negative_indices = sampled_negative_indices.to(device)

        outputs = model(
            input_values=input_values,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices,
        )

        loss = outputs.loss
        loss.backward()
        grad_norm.append(get_grad_norm(model))
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    avg_grad = sum(grad_norm) / len(grad_norm)
    return avg_loss, avg_grad


def validate(
    model: Wav2Vec2ForPreTraining, val_loader: DataLoader, device: torch.device
) -> float:
    """Compute SSL validation loss for wav2vec2 pretraining."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (input_values, labels) in val_loader:
            input_values = input_values.float().to(device)
            mask_time_indices, sampled_negative_indices = compute_mask_inputs(
                model, input_values, device
            )
            outputs = model(
                input_values=input_values,
                mask_time_indices=mask_time_indices,
                sampled_negative_indices=sampled_negative_indices,
            )
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train_probe(
    model: Wav2Vec2ForPreTraining,
    train_loader: DataLoader,
    val_loader: DataLoader,
    rep_dim: int,
    num_classes: int,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Train a frozen-encoder linear probe for quick representation quality checks."""
    for p in model.parameters():
        p.requires_grad = False

    prober = LinearProber(model, rep_dim, num_classes).to(device)
    prober.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(prober.parameters(), lr=5e-6)
    train_loss, train_correct, val_loss, val_correct = 0, 0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = prober(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        train_correct += (logits.argmax(1) == yb).sum().item()

    prober.eval()
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = prober(xb)
        loss = criterion(logits, yb)
        val_loss += loss.item() * xb.size(0)
        val_correct += (logits.argmax(1) == yb).sum().item()

    train_avg_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)
    val_avg_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    for p in model.parameters():
        p.requires_grad = True

    return train_avg_loss, train_acc, val_avg_loss, val_acc
