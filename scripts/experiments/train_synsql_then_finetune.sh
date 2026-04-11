#!/bin/bash
# Section 7.2: Stage 2 — Finetune a SynSQL-pretrained checkpoint on Spider or BIRD
#
# This two-stage recipe (SynSQL pretrain → benchmark finetune) is used for all
# SoTA comparison and extended benchmark experiments.
# Adjust PRETRAINED_CKPT to the best checkpoint from Stage 1.

cd "$(dirname "$0")/../../training"

PRETRAINED_CKPT="ra_sql_ckpts/ra_sql_sft_synsql_10p/checkpoint-100000"

# --- Finetune on Spider ---
python train.py \
    --model_name_or_path ${PRETRAINED_CKPT} \
    --max_length 12800 \
    --batch_size 1 \
    --num_epochs 3 \
    --database_path ../data/spider/database \
    --dataset_name spider \
    --table_value_cache_path ../data/spider/spider_train_id2sampled_values.json \
    --table_info_cache_path ../data/spider/spider_train_id2db_info.json \
    --input_file ../data/spider/train_spider_ra_correct.json \
    --use_wandb \
    --task_type ra_sql \
    --output_dir ra_sql_ckpts/ra_sql_synsql_then_spider \
    --eval_steps 1000

# --- Finetune on BIRD ---
# python train.py \
#     --model_name_or_path ${PRETRAINED_CKPT} \
#     --max_length 12800 \
#     --batch_size 1 \
#     --num_epochs 3 \
#     --database_path ../data/bird/train/train_databases \
#     --dataset_name bird \
#     --table_value_cache_path ../data/bird/train/bird_train_id2sampled_values.json \
#     --table_info_cache_path ../data/bird/train/bird_train_id2db_info.json \
#     --input_file ../data/bird/train/train_bird_ra_correct.json \
#     --use_wandb \
#     --task_type ra_sql \
#     --output_dir ra_sql_ckpts/ra_sql_synsql_then_bird \
#     --eval_steps 1000
