"""
run_pipeline.py
===============
Test Runner Script for Project Fractal.
Executes the factory slicing pipeline on a downloaded local model checkpoint
and saves the output flatbuffer segments (.pte) to the output directory.

Usage:
------
python run_pipeline.py --model_path ./assets/raw_models/Meta-Llama-3-8B --layers 32 --output_dir ./output
"""

import argparse
import os
import sys

# Ensure the parent directory is in the import search path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from slicer.pipeline import run_factory_pipeline
except ImportError as e:
    print(f"[ERROR] Failed to import slicer module: {e}")
    print("Ensure you run this script from the root directory of the tool.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fractal Conveyor Slicing Pipeline Test Runner")
    parser.add_argument(
        "--model_path",
        type=str,
        default="T:\\models\\Meta-Llama-3-8B",
        help="Local filesystem path to the model checkpoint folder (or Hugging Face ID if online)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=32,
        help="Number of layers to slice/process (default: 32)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the resulting .pte flatbuffer layers",
    )
    args = parser.parse_args()

    # Convert paths to absolute for clarity
    abs_model_path = os.path.abspath(args.model_path)
    abs_output_dir = os.path.abspath(args.output_dir)

    print(f"\n{'='*60}")
    print("  FRACTAL FACTORY RUNNER")
    print(f"  Source Model Checkpoint: {abs_model_path}")
    print(f"  Layer Processing Target: {args.layers} layers")
    print(f"  Target Output Location:  {abs_output_dir}")
    print(f"{'='*60}\n")

    if not os.path.exists(abs_model_path):
        print(f"[WARN] Local directory '{abs_model_path}' does not exist.")
        print("We will attempt to load directly via Hugging Face Hub (requires network/credentials).")
        print("To download the model locally first, run: python slicer/scripts/00_download_model.py")
        print("-" * 60)

    try:
        run_factory_pipeline(
            model_path=args.model_path,
            total_layers=args.layers,
            output_dir=abs_output_dir,
        )

        print(f"\n[OK] Run completed successfully.")
        print(f"Outputs written to: {abs_output_dir}")
        print("\nGenerated files:")
        if os.path.exists(abs_output_dir):
            for file in sorted(os.listdir(abs_output_dir)):
                if file.endswith(".pte"):
                    full_path = os.path.join(abs_output_dir, file)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    print(f"  - {file} ({size_mb:.2f} MB)")
        else:
            print("  No output files found.")

    except Exception as exc:
        print(f"\n[CRITICAL ERROR] Factory Pipeline crashed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
