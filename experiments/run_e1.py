"""Convenience wrapper for primary E1."""
import argparse
from experiments.datasets import DATASETS
from experiments.run_primary import run_primary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="cifar10")
    parser.add_argument("--pilot-epochs", type=int, default=None)
    args = parser.parse_args()
    run_primary("E1", args.dataset, pilot_epochs=args.pilot_epochs)


if __name__ == "__main__":
    main()
