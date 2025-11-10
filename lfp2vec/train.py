import gc
import logging
import os
import pickle
import sys
import tempfile
from typing import Tuple
from uuid import uuid4

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functools import partial
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
    upsample_collate,
    upsample_collate_for_trainer,
)

from blind_localization.data.PCAviz import PCAVisualizer
from blind_localization.models.brainbert import (
    BrainBERTModel,
    PretrainMaskedCriterion,
    train as brainbert_train_epoch,
    validation as brainbert_validate_epoch,
    BrainBERTWithMLP,
    train_decoder as brainbert_train_decoder,
    validate_decoder as brainbert_validate_decoder,
)
from blind_localization.data.datasets import RawDataset
from blind_localization.models.contrastive_pipeline import (
    train as simclr_train_epoch,
    validation as simclr_validate_epoch,
)
from blind_localization.models.contrastive import ContrastiveEncoder, InfoNCELoss
from blind_localization.models.decoder import (
    ContrastiveLearningWithLR,
    ContrastiveLearningWithMLP,
    train_decoder as simclr_train_decoder,
    validate_decoder as simclr_validate_decoder,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training(
    data: str = "Allen",
    data_type: str = "spectrogram_preprocessed",
    train_session_size: float = 0.8,
    trial_length: int = 60,
    onthefly_upsample: bool = True,
    sampling_rate: int = 1250,
    rand_init: bool = False,
    ssl_method: str = "lfp2vec",
    epoch: int = 50,
    lr: float = 1e-5,
):
    """End-to-end pipeline: pretraining (optional), embedding viz, fine-tuning, logging.

    - Builds datasets, optionally does wav2vec2 SSL pretraining, visualizes classifier-input
      embeddings pre/post FT, fine-tunes classification head, and saves results.
    """

    logger.critical("[RUN_TRAINING] Running training with the following parameters:")
    logger.critical(f"GPU: {torch.cuda.is_available()}")
    logger.critical(f"Data: {data}")
    logger.critical(f"Data type: {data_type}")
    logger.critical(f"On-the-fly upsampling: {onthefly_upsample}")
    logger.critical(f"Sampling rate: {sampling_rate}")
    logger.critical(f"Random initialization: {rand_init}")
    logger.critical(f"SSL method: {ssl_method}")
    logger.critical(f"Epoch: {epoch}")
    logger.critical(f"Learning rate: {lr}")

    # tags
    ri_tag = "rand_init" if rand_init else "pretrained"
    ssl_tag = ssl_method
    ft_tag = "no_ft"

    # output path
    output_path = f"../results/{data}/{data_type}/wave2vec2/across_session"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # load data
    data_loader = LFP2VecDataLoader(
        data, train_session_size=train_session_size, trial_length=trial_length
    )
    train_dataset, val_dataset, test_dataset = data_loader.parse_datasets(
        sampling_rate=None if onthefly_upsample else sampling_rate
    )

    logger.info(
        f"Train sessions: {data_loader.train_sess}, "
        f"Validation sessions: {data_loader.val_sess}, "
        f"Test sessions: {data_loader.test_sess}"
    )
    logger.info(
        f"Train trials: {data_loader.train_trials}, "
        f"Validation trials: {data_loader.val_trials}, "
        f"Test trials: {data_loader.test_trials}"
    )
    # id2label, label2id
    acronyms_arr = data_loader.hc_acronyms
    id2label = {str(i): acr for i, acr in enumerate(acronyms_arr)}
    label2id = {acr: str(i) for i, acr in enumerate(acronyms_arr)}

    logger.info(f"label2id: {label2id}")
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
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
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
        self_supervised=ssl_method == "lfp2vec",
    )

    # Ensure classification head label space is correct
    w2v2_config.num_labels = len(id2label)
    w2v2_config.label2id = label2id
    w2v2_config.id2label = id2label

    # convert to dict
    w2v2_config_dict = w2v2_config.to_dict()

    logger.info("Initializing WandB...")
    wandb.init(
        project="lfp2vec",
        config=w2v2_config_dict,
        name=(f"{data}-" f"{ri_tag}-" f"{ssl_tag}-exp"),
        reinit=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ssl_method == "lfp2vec":
        logger.info("Initializing LFP2Vec Model...")
        if rand_init:
            ssl_model = Wav2Vec2ForPreTraining(config=w2v2_config)
        else:
            ssl_model = Wav2Vec2ForPreTraining.from_pretrained(
                "facebook/wav2vec2-base",
                config=w2v2_config,
                ignore_mismatched_sizes=True,
            )

        ssl_model.quantizer.register_forward_hook(quantizer_hook)
        ssl_model.to(device)
        logger.info(f"Model is on device: {ssl_model.device}")

        logger.info("Training LFP2Vec Model...")
        optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=lr)

        collate = partial(
            upsample_collate,
            target_sampling_rate=16000,
            source_sampling_rate=sampling_rate,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate if onthefly_upsample else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate if onthefly_upsample else None,
        )
        # test_loader not needed for SSL training loop here
        max_probe_acc = 0

        for epoch in tqdm(range(epoch)):
            train_loss, grad_norm = train(ssl_model, train_loader, optimizer, device)
            val_loss = validate(ssl_model, val_loader, device)
            logger.info(
                f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Grad Norm: {grad_norm:.4f}"
            )
            wandb.log(
                {
                    "ssl/train_loss": train_loss,
                    "ssl/val_loss": val_loss,
                    "ssl/grad_norm": grad_norm,
                    "ssl/lr": optimizer.param_groups[0]["lr"],
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
                        "probe/train_loss": probe_train_loss,
                        "probe/train_acc": probe_train_acc,
                        "probe/val_loss": probe_val_loss,
                        "probe/val_acc": probe_val_acc,
                    }
                )
                if max_probe_acc > probe_val_acc:
                    max_probe_acc = max(max_probe_acc, probe_val_acc)
                    ssl_model.save_pretrained(f"{output_path}/disease/ssl_model/")

        # ================================ Training phase lfp2vec ================================
        model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base",
            config=w2v2_config,
            ignore_mismatched_sizes=True,
        )

        # Check if checkpoint exists
        if os.path.exists(f"{output_path}/disease/ssl_model/"):
            ssl_model = Wav2Vec2ForPreTraining.from_pretrained(
                f"{output_path}/disease/ssl_model/",
                ignore_mismatched_sizes=True,
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

        uuid = uuid4().hex
        unique_cache_dir = tempfile.mkdtemp(prefix="hf_eval_")
        accuracy = evaluate.load(
            "accuracy", experiment_id=str(uuid), cache_dir=unique_cache_dir
        )
        f1_metric = evaluate.load(
            "f1", experiment_id=str(uuid) + "_f1", cache_dir=unique_cache_dir
        )

        def compute_metrics(eval_pred):
            """Compute accuracy from Trainer eval predictions."""
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return accuracy.compute(
                predictions=predictions, references=eval_pred.label_ids
            )

        # Setup trainer (used later for fine-tuning)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=upsample_collate_for_trainer,
            compute_metrics=compute_metrics,
        )

        # Register hooks
        projector_hook_handle = model.projector.register_forward_hook(projector_hook)
        classifier_hook_handle = model.classifier.register_forward_hook(classifier_hook)

        # Prepare for embedding collection before fine-tuning
        model.to(device)
        train_eval_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate if onthefly_upsample else None,
        )
        val_eval_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate if onthefly_upsample else None,
        )
        test_eval_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate if onthefly_upsample else None,
        )

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
                val_embeddings,
                val_labels,
                2,
                f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
                "val",
            )
            visualizer.create_pca(
                val_embeddings,
                val_labels,
                3,
                f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
                "val",
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
        train_pred = trainer.predict(train_dataset)
        train_logits = train_pred[0]
        train_labels = train_pred[1]
        train_acc = train_pred[2]["test_accuracy"]
        train_preds = np.argmax(train_logits, axis=1)
        train_f1 = f1_metric.compute(
            predictions=train_preds, references=train_labels, average="macro"
        )["f1"]

        val_pred = trainer.predict(val_dataset)
        val_logits = val_pred[0]
        val_labels = val_pred[1]
        val_acc = val_pred[2]["test_accuracy"]
        val_preds = np.argmax(val_logits, axis=1)
        val_f1 = f1_metric.compute(
            predictions=val_preds, references=val_labels, average="macro"
        )["f1"]

        test_pred = trainer.predict(test_dataset)
        test_logits = test_pred[0]
        test_labels = test_pred[1]
        test_acc = test_pred[2]["test_accuracy"]
        test_preds = np.argmax(test_logits, axis=1)
        test_f1 = f1_metric.compute(
            predictions=test_preds, references=test_labels, average="macro"
        )["f1"]

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
                val_embeddings,
                val_labels,
                2,
                f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
                "val",
            )
            visualizer.create_pca(
                val_embeddings,
                val_labels,
                3,
                f"{data}_{ri_tag}_{ssl_tag}_{ft_tag}",
                "val",
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
                "train/accuracy": train_acc,
                "train/f1": train_f1,
                "val/accuracy": val_acc,
                "val/f1": val_f1,
                "test/accuracy": test_acc,
                "test/f1": test_f1,
            }
        )

        # Deregister hooks
        projector_hook_handle.remove()
        classifier_hook_handle.remove()
        gc.collect()
        torch.cuda.empty_cache()
        return

    elif ssl_method == "brainbert":
        logger.info("Preparing BrainBERT spectrogram datasets and dataloaders...")
        # Align data pipeline to BrainBERT script using contrastive_pipeline builders
        spectrogram_size = 500
        time_bins = 16
        batch_size = 32

        bb_train_dataset = RawDataset(
            train_dataset.data,
            train_dataset.labels,
            spectrogram_size=spectrogram_size,
            time_bins=time_bins,
            library="pytorch",
        )
        bb_val_dataset = RawDataset(
            val_dataset.data,
            val_dataset.labels,
            spectrogram_size=spectrogram_size,
            time_bins=time_bins,
            library="pytorch",
        )
        bb_test_dataset = RawDataset(
            test_dataset.data,
            test_dataset.labels,
            spectrogram_size=spectrogram_size,
            time_bins=time_bins,
            library="pytorch",
        )

        bb_train_loader = DataLoader(
            bb_train_dataset, batch_size=batch_size, shuffle=True
        )
        bb_val_loader = DataLoader(bb_val_dataset, batch_size=batch_size, shuffle=True)
        bb_test_loader = DataLoader(
            bb_test_dataset, batch_size=batch_size, shuffle=False
        )

        class _BBCfg:
            pass

        cfg = _BBCfg()
        cfg.input_dim = spectrogram_size
        cfg.hidden_dim = 8 * 16  # nhead * hid_dim_factor
        cfg.layer_dim_feedforward = 512
        cfg.layer_activation = "gelu"
        cfg.nhead = 8
        cfg.encoder_num_layers = 4

        logger.info("Initializing BrainBERT model...")
        brainbert_model = BrainBERTModel()
        brainbert_model.build_model(cfg)
        brainbert_model.to(device)

        criterion = PretrainMaskedCriterion(alpha=2)
        optimizer = torch.optim.Adam(brainbert_model.parameters(), lr=lr)

        best_val = float("inf")
        for ep in range(epoch):
            tr_loss = brainbert_train_epoch(
                brainbert_model, bb_train_loader, optimizer, criterion, device
            )
            va_loss = brainbert_validate_epoch(
                brainbert_model, bb_val_loader, criterion, device
            )
            logger.info(
                f"[BrainBERT] Epoch {ep+1}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f}"
            )
            wandb.log(
                {"bb_train_loss": tr_loss, "bb_val_loss": va_loss, "bb_epoch": ep + 1}
            )
            if va_loss <= best_val:
                best_val = va_loss
                os.makedirs(f"{output_path}/disease", exist_ok=True)
                torch.save(
                    brainbert_model.state_dict(),
                    f"{output_path}/disease/brainbert_ssl.pt",
                )

        # Load best SSL checkpoint for feature-extractor baseline
        ckpt_path = f"{output_path}/disease/brainbert_ssl.pt"
        if os.path.exists(ckpt_path):
            brainbert_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            logger.info(f"Loaded BrainBERT SSL checkpoint: {ckpt_path}")

        # Build frozen-encoder + MLP classifier (feature extractor baseline)
        input_size = bb_train_loader.dataset[0][0][0].size()[0]
        input_dim = (
            input_size // spectrogram_size
        ) * cfg.hidden_dim  # time_bins * hidden_dim
        clf_model = BrainBERTWithMLP(
            brainbert_model,
            input_dim=input_dim,
            hidden_dim=256,
            output_dim=len(id2label),
        ).to(device)

        combined_criterion = nn.CrossEntropyLoss()
        combined_optimizer = torch.optim.Adam(clf_model.parameters(), lr=3e-5)

        best_val_acc = 0.0
        for ep in range(12):
            tr = brainbert_train_decoder(
                clf_model,
                bb_train_loader,
                combined_optimizer,
                criterion,
                combined_criterion,
                device=device,
                mode="separate",
            )
            va = brainbert_validate_decoder(
                clf_model, bb_val_loader, combined_criterion, device=device
            )
            # Support both tuple and dict return signatures
            if isinstance(tr, dict):
                dec_train_loss = float(tr.get("loss", 0.0))
                dec_train_acc = float(tr.get("balanced_accuracy", 0.0))
            else:
                dec_train_loss, dec_train_acc = tr
            if isinstance(va, dict):
                dec_val_loss = float(va.get("loss", 0.0))
                dec_val_acc = float(va.get("balanced_accuracy", 0.0))
                dec_val_f1 = float(va.get("macro_f1", 0.0))
            else:
                dec_val_loss, dec_val_acc, dec_val_f1 = va

            wandb.log(
                {
                    "decoder/train_loss": dec_train_loss,
                    "decoder/train_acc": dec_train_acc,
                    "decoder/val_loss": dec_val_loss,
                    "decoder/val_acc": dec_val_acc,
                    "decoder/val_f1": dec_val_f1,
                    "decoder/epoch": ep + 1,
                }
            )
            logger.info(
                f"[BrainBERT-Decoder] Epoch {ep+1}: train_loss={dec_train_loss:.4f} "
                f"train_acc={dec_train_acc:.4f} val_loss={dec_val_loss:.4f} "
                f"val_acc={dec_val_acc:.4f} val_f1={dec_val_f1:.4f}"
            )
            best_val_acc = max(best_val_acc, dec_val_acc)

        # Final test evaluation
        test_metrics = brainbert_validate_decoder(
            clf_model, bb_test_loader, combined_criterion, device=device
        )
        if isinstance(test_metrics, dict):
            test_acc = float(test_metrics.get("balanced_accuracy", 0.0))
            test_f1 = float(test_metrics.get("macro_f1", 0.0))
        else:
            _, test_acc, test_f1 = test_metrics
        logger.info(
            f"[BrainBERT-Decoder] Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}"
        )
        wandb.log(
            {
                "ssl_method": "brainbert",
                "test/accuracy": test_acc,
                "test/f1": test_f1,
            }
        )

        # Save minimal results file for BrainBERT baseline
        file_path = os.path.join(
            output_path, f"{data}_{ri_tag}_brainbert_ft_results.pickle"
        )
        file_obj = {
            "val_best_acc": best_val_acc,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "bb_cfg": {
                "spectrogram_size": spectrogram_size,
                "time_bins": time_bins,
                "hidden_dim": cfg.hidden_dim,
                "nhead": cfg.nhead,
                "encoder_num_layers": cfg.encoder_num_layers,
            },
        }
        with open(file_path, "wb") as f:
            pickle.dump(file_obj, f)
        logger.info(f"BrainBERT baseline results saved to {file_path}")

        # Baseline finished, skip wav2vec2 fine-tuning path
        gc.collect()
        torch.cuda.empty_cache()
        return

    elif ssl_method in ("simclr", "simclr_mlp"):
        logger.info("Preparing SimCLR dataloaders and model...")
        # Use across-session split like lfp2vec
        spectrogram_size = 500
        time_bins = 16
        batch_size = 64

        # Build RawDataset-based loaders using the same aggregated sets
        sim_train_ds = RawDataset(
            train_dataset.data,
            train_dataset.labels,
            spectrogram_size=spectrogram_size,
            time_bins=time_bins,
            library="pytorch",
        )
        sim_val_ds = RawDataset(
            val_dataset.data,
            val_dataset.labels,
            spectrogram_size=spectrogram_size,
            time_bins=time_bins,
            library="pytorch",
        )
        sim_test_ds = RawDataset(
            test_dataset.data,
            test_dataset.labels,
            spectrogram_size=spectrogram_size,
            time_bins=time_bins,
            library="pytorch",
        )

        sim_train_loader = DataLoader(sim_train_ds, batch_size=batch_size, shuffle=True)
        sim_val_loader = DataLoader(sim_val_ds, batch_size=batch_size, shuffle=True)
        sim_test_loader = DataLoader(sim_test_ds, batch_size=batch_size, shuffle=False)

        input_size = sim_train_loader.dataset[0][0][0].size()[0]

        # Defaults inspired by simclr_lr_decoder.py (compact for baseline)
        fc_layer_size = 256
        latent_size = 128
        temperature = 0.5
        encoder_epochs = epoch
        decoder_epochs = 20

        encoder = ContrastiveEncoder(
            fc_layer_size=fc_layer_size,
            input_size=input_size,
            output_size=latent_size,
        ).to(device)

        if ssl_method == "simclr_mlp":
            sim_model = ContrastiveLearningWithMLP(
                encoder,
                input_dim=latent_size,
                hidden_dim=256,
                output_dim=len(id2label),
            ).to(device)
        else:
            sim_model = ContrastiveLearningWithLR(
                encoder,
                input_dim=latent_size,
                output_dim=len(id2label),
            ).to(device)

        contrastive_criterion = InfoNCELoss(temperature=temperature, device=device)
        supervised_criterion = nn.CrossEntropyLoss()
        enc_opt = torch.optim.Adam(sim_model.encoder.parameters(), lr=lr)
        dec_opt = torch.optim.Adam(sim_model.parameters(), lr=3e-5)

        # Stage 1: Unsupervised pretraining
        for ep in range(encoder_epochs):
            tr_loss = simclr_train_epoch(
                sim_model.encoder,
                sim_train_loader,
                enc_opt,
                contrastive_criterion,
                device,
            )
            va_loss = simclr_validate_epoch(
                sim_model.encoder, sim_val_loader, contrastive_criterion, device
            )
            logger.info(
                f"[SimCLR] Epoch {ep+1}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f}"
            )
            wandb.log(
                {
                    "ssl_method": "simclr",
                    "ssl/train_loss": tr_loss,
                    "ssl/val_loss": va_loss,
                    "ssl/epoch": ep + 1,
                }
            )

        # Stage 2: Supervised decoder on frozen or joint (use separate/frozen)
        best_val_acc = 0.0
        patience = 10
        patience_ctr = 0
        for ep in range(decoder_epochs):
            tr = simclr_train_decoder(
                sim_model,
                sim_train_loader,
                dec_opt,
                contrastive_criterion,
                supervised_criterion,
                mode="separate",
                device=device,
            )
            va = simclr_validate_decoder(
                sim_model, sim_val_loader, supervised_criterion, device=device
            )
            # tolerant to tuple/dict signatures
            if isinstance(tr, dict):
                dec_train_loss = float(tr.get("loss", 0.0))
                dec_train_acc = float(
                    tr.get("balanced_accuracy", tr.get("accuracy", 0.0))
                )
            else:
                dec_train_loss, dec_train_acc = tr
            if isinstance(va, dict):
                dec_val_loss = float(va.get("loss", 0.0))
                dec_val_acc = float(va.get("balanced_accuracy", 0.0))
                dec_val_f1 = float(va.get("macro_f1", 0.0))
            else:
                dec_val_loss, dec_val_acc, dec_val_f1 = va

            wandb.log(
                {
                    "ssl_method": "simclr",
                    "decoder/train_loss": dec_train_loss,
                    "decoder/train_acc": dec_train_acc,
                    "decoder/val_loss": dec_val_loss,
                    "decoder/val_acc": dec_val_acc,
                    "decoder/val_f1": dec_val_f1,
                    "decoder/epoch": ep + 1,
                }
            )
            logger.info(
                f"[SimCLR-Decoder] Epoch {ep+1}: train_loss={dec_train_loss:.4f} "
                f"train_acc={dec_train_acc:.4f} val_loss={dec_val_loss:.4f} "
                f"val_acc={dec_val_acc:.4f} val_f1={dec_val_f1:.4f}"
            )
            if dec_val_acc > best_val_acc:
                best_val_acc = dec_val_acc
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

        # Final test metrics
        test_metrics = simclr_validate_decoder(
            sim_model, sim_test_loader, supervised_criterion, device=device
        )
        if isinstance(test_metrics, dict):
            sim_test_acc = float(test_metrics.get("balanced_accuracy", 0.0))
            sim_test_f1 = float(test_metrics.get("macro_f1", 0.0))
        else:
            _, sim_test_acc, sim_test_f1 = test_metrics

        logger.info(
            f"[SimCLR-Decoder] Test Acc={sim_test_acc:.4f}, Test F1={sim_test_f1:.4f}"
        )
        wandb.log(
            {
                "ssl_method": "simclr",
                "test/accuracy": sim_test_acc,
                "test/f1": sim_test_f1,
            }
        )

        # Save compact results
        file_path = os.path.join(
            output_path, f"{data}_{ri_tag}_simclr_ft_results.pickle"
        )
        file_obj = {
            "val_best_acc": best_val_acc,
            "test_acc": sim_test_acc,
            "test_f1": sim_test_f1,
            "simclr_cfg": {
                "fc_layer_size": fc_layer_size,
                "latent_size": latent_size,
                "temperature": temperature,
                "encoder_epochs": encoder_epochs,
                "decoder_epochs": decoder_epochs,
                "batch_size": batch_size,
                "spectrogram_size": spectrogram_size,
                "time_bins": time_bins,
            },
        }
        with open(file_path, "wb") as f:
            pickle.dump(file_obj, f)
        logger.info(f"SimCLR baseline results saved to {file_path}")

        gc.collect()
        torch.cuda.empty_cache()
        return


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
    for input_values, labels in train_loader:
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
        for input_values, labels in val_loader:
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
