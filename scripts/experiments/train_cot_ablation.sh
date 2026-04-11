#!/bin/bash
# Section 7.5: CoT ablation — compare RA-SQL CoT vs direct SQL (no CoT)
#
# Both variants are trained on the same SynSQL 10% data with the same model,
# differing only in whether RA-based CoT is included in the training target.
# Evaluation uses the same eval scripts as Section 7.2.

cd "$(dirname "$0")/../../training"

# --- Variant 1: RA-SQL (with RA-based CoT) ---
# Already trained in train_synsql_pretrain.sh with task_type=ra_sql

# --- Variant 2: SQL-only baseline (no CoT) ---
python train.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-3B-Instruct \
    --cached_dataset_path ../data/synsql/cached_datasets/cached_dataset.pkl \
    --max_length 12800 \
    --batch_size 1 \
    --num_epochs 1 \
    --use_wandb \
    --task_type sql \
    --output_dir ra_sql_ckpts/sql_sft_synsql_10p_no_cot \
    --eval_steps 10000 \
    --save_limit 20

# After training both variants, evaluate with the same eval commands
# (scripts/experiments/eval_extended_benchmarks.sh), changing MODEL_CHECKPOINT
# to point to each variant's checkpoint.
