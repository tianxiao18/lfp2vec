import argparse

from train import run_training


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running training pipeline."""
    parser = argparse.ArgumentParser(description="Run lfp2vec training pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="Allen",
        choices=["Allen", "ibl", "Neuronexus", "Monkey", "All"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="spectrogram_preprocessed",
        help="Data type tag used in result path",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="train session split proportion (0-1)",
    )
    parser.add_argument(
        "--trial_length",
        type=int,
        default=60,
        help="Total number of trials",
    )
    parser.add_argument(
        "--onthefly_upsample",
        type=bool,
        default=True,
        help="Upsample data to 16 kHz on the fly",
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
        train_session_size=args.train_size,
        trial_length=args.trial_length,
        onthefly_upsample=args.onthefly_upsample,
        sampling_rate=args.sampling_rate,
        rand_init=args.rand_init,
        ssl=args.ssl,
        epoch=args.epoch,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
