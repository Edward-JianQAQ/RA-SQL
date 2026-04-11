#!/bin/bash
# Train an RA-SQL model using supervised fine-tuning.
#
# This script demonstrates training on the BIRD dataset with RA+SQL CoT.
# Adjust paths and hyperparameters as needed.

cd "$(dirname "$0")/../training"

# RA+SQL training on BIRD (single GPU)
python train.py \
  --model_name_or_path Qwen/Qwen2.5-Coder-3B-Instruct \
  --task_type ra_sql \
  --database_path ../data/bird/train/train_databases \
  --dataset_name bird \
  --table_value_cache_path ../data/bird/train/bird_train_id2sampled_values.json \
  --table_info_cache_path ../data/bird/train/bird_train_id2db_info.json \
  --input_file ../data/bird/train/train_bird_ra_correct.json \
  --max_length 4096 \
  --batch_size 4 \
  --num_epochs 3 \
  --cot \
  --serialize_cot_type post_order \
  --output_dir ra_sql_sft_model \
  --save_limit 3 \
  --use_wandb

# For multi-GPU training with DeepSpeed ZeRO-3:
# deepspeed --num_gpus 4 train.py \
#   --deepspeed \
#   --deepspeed_config ds_config_zero3.json \
#   ... (same args as above)
