"""Command-line interface for running AES CPA on a dataset folder."""

import argparse
from pathlib import Path

from .cpa import recover_key

def run_cli():
    parser = argparse.ArgumentParser(description="Run CPA key recovery for AES dataset")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="path to dataset directory")
    parser.add_argument("--output-csv", type=Path, required=True, help="where to save per-byte CPA scores",)
    parser.add_argument('--clock', action='store_true', help="set whether the clock traces are present")
    args = parser.parse_args()

    key, output_dataset = recover_key(args.dataset_dir, args.clock)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_dataset.to_csv(args.output_csv, index=False)

    key_hex = " ".join(f"{b:02x}" for b in key)
    print(f"Recovered key: {bytes(key)}")
    print(f"Recovered key (hex): {key_hex}")
    print(f"Recovered key byte sum (decimal): {int(key.sum())}")
    print(f"Saved per-byte results to: {args.output_csv}")

if __name__ == "__main__":
    run_cli()
