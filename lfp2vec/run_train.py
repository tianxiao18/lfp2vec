import argparse

from train import run_training


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running training pipeline."""
    parser = argparse.ArgumentParser(description="Run lfp2vec training pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="Allen",
        choices=["Allen", "ibl", "Neuronexus", "All"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="spectrogram_preprocessed",
        help="Data type tag used in result path",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Validation split proportion (0-1)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split proportion (0-1)",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=1250,
        help="Target sampling rate for upsampling (Hz)",
    )
    parser.add_argument(
        "--rand_init",
        action="store_true",
        help="Initialize wav2vec2 randomly instead of loading pretrained",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="Enable self-supervised pretraining before fine-tuning",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="Number of SSL pretraining epochs (used when --ssl)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for SSL pretraining optimizer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        data=args.data,
        data_type=args.data_type,
        val_size=args.val_size,
        test_size=args.test_size,
        sampling_rate=args.sampling_rate,
        rand_init=args.rand_init,
        ssl=args.ssl,
        epoch=args.epoch,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

