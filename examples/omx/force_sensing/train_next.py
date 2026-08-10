#!/usr/bin/env python3
"""Train a FACTR2 NEXT external-torque estimator (arXiv:2606.12406) from free-motion logs
produced by `collect_free_motion.py`.

Usage (run from repo root):
    python -m examples.omx.force_sensing.train_next \\
        --data data/omx_free_motion/run1.npz data/omx_free_motion/run2.npz \\
        --output checkpoints/omx_next.pt
"""

import argparse
import logging

from lerobot.force_estimation import NextTrainConfig, train_next


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Path(s) to .npz free-motion logs from collect_free_motion.py",
    )
    parser.add_argument("--output", required=True, help="Output checkpoint path (.pt)")
    parser.add_argument("--history-length", type=int, default=50)
    parser.add_argument("--resample-hz", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = NextTrainConfig(
        data_paths=args.data,
        output_path=args.output,
        history_length=args.history_length,
        resample_hz=args.resample_hz,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_next(cfg)


if __name__ == "__main__":
    main()
