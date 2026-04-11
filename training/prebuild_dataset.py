#!/usr/bin/env python3
"""
Pre-build and cache the dataset to avoid slow loading on every training run.
This script builds the full dataset once and saves it in a fast-loading format.
"""
import argparse
import json
import os
import pickle
import torch
from transformers import AutoTokenizer
from train import ChatAlignedRelationalQueryDataset

def prebuild_dataset(args):
    """Pre-build dataset and save to disk."""
    print("=" * 80)
    print("Pre-building dataset for fast loading...")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Load input data
    print(f"\nLoading input data from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        input_dataset = json.load(f)
    print(f"Loaded {len(input_dataset)} samples")

    # Build dataset (this is the slow part we want to do once)
    print(f"\nBuilding dataset (this may take several minutes)...")
    print(f"  Task type: {args.task_type}")
    print(f"  CoT: {args.cot}")
    print(f"  Serialize type: {args.serialize_cot_type}")

    dataset = ChatAlignedRelationalQueryDataset(
        input_dataset=input_dataset,
        database_path=args.database_path,
        dataset_name=args.dataset_name,
        table_value_cache_path=args.table_value_cache_path,
        table_info_cache_path=args.table_info_cache_path,
        tokenizer=tokenizer,
        task_type=args.task_type,
        max_length=args.max_length,
        mode='train',
        cot=args.cot,
        serialize_cot_type=args.serialize_cot_type,
        full_loss=args.full_loss,
        sel_seed=args.sel_seed,
        apply_template=args.apply_template,
    )

    print(f"\n✓ Dataset built successfully with {len(dataset)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save dataset
    output_path = os.path.join(args.output_dir, "cached_dataset.pkl")
    print(f"\nSaving pre-built dataset to {output_path}...")

    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
    metadata = {
        'num_samples': len(dataset),
        'task_type': args.task_type,
        'max_length': args.max_length,
        'cot': args.cot,
        'serialize_cot_type': args.serialize_cot_type,
        'model_name_or_path': args.model_name_or_path,
        'input_file': args.input_file,
        'dataset_name': args.dataset_name,
    }
    metadata_path = os.path.join(args.output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Pre-built dataset saved!")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Metadata: {metadata_path}")
    print(f"\nTo use this pre-built dataset, add this flag to training:")
    print(f"  --cached_dataset_path={output_path}")

def main():
    parser = argparse.ArgumentParser(description="Pre-build dataset for fast loading")

    # Model args
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Model name or path for tokenizer")

    # Data args
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSON file")
    parser.add_argument("--database_path", type=str, required=True,
                        help="Path to database directory")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name (e.g., spider, bird, synsql)")
    parser.add_argument("--table_value_cache_path", type=str, required=True,
                        help="Path to table value cache JSON")
    parser.add_argument("--table_info_cache_path", type=str, required=True,
                        help="Path to table info cache JSON")

    # Task args
    parser.add_argument("--task_type", type=str, default="ra_sql",
                        choices=["ra", "sql", "ra_sql"],
                        help="Task type")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument("--cot", action="store_true",
                        help="Use chain-of-thought")
    parser.add_argument("--serialize_cot_type", type=str, default="",
                        help="CoT serialization method")
    parser.add_argument("--full_loss", action="store_true",
                        help="Compute loss on full sequence")
    parser.add_argument("--sel_seed", type=int, default=None,
                        help="Random seed for serialization")
    parser.add_argument("--apply_template", action="store_true",
                        help="Apply chat template")

    # Output args
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save pre-built dataset")

    args = parser.parse_args()

    prebuild_dataset(args)

if __name__ == "__main__":
    main()
