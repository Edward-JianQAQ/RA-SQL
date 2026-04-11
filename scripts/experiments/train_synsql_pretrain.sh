#!/bin/bash
# Section 7.2: Stage 1 — Pretrain on SynSQL subset (10% of 2.5M)
# This produces a base checkpoint used for downstream finetuning on Spider/BIRD.
#
# Requires: cached SynSQL dataset (see README for preparation instructions)

cd "$(dirname "$0")/../../training"

# RA-SQL pretraining on SynSQL 10%
python train.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-3B-Instruct \
    --cached_dataset_path ../data/synsql/cached_datasets/cached_dataset.pkl \
    --max_length 12800 \
    --batch_size 1 \
    --num_epochs 3 \
    --use_wandb \
    --task_type ra_sql \
    --output_dir ra_sql_ckpts/ra_sql_sft_synsql_10p \
    --eval_steps 10000 \
    --save_limit 15
