#!/bin/bash
# Evaluate an RA-SQL model using vLLM for batch inference.
#
# Adjust the checkpoint path and data paths to match your setup.

cd "$(dirname "$0")/../evaluation"

# RA+SQL evaluation on BIRD-dev
python eval.py \
  --model_checkpoint_path ../training/ra_sql_ckpts/ra_sql_sft_model \
  --tokenizer_name Qwen/Qwen2.5-Coder-3B-Instruct \
  --task_type ra_sql \
  --database_path ../data/bird/dev/dev_databases \
  --dataset_name bird \
  --table_value_cache_path ../data/bird/dev/bird_dev_id2sampled_values.json \
  --table_info_cache_path ../data/bird/dev/bird_dev_id2db_info.json \
  --input_file ../data/bird/dev/dev_bird_ra_correct.json \
  --cot \
  --tensor_parallel_size 1 \
  --batch_size 32 \
  --detailed_log eval_bird_dev.txt \
  --save_errors

# SQL evaluation on Spider-dev (with foreign key support)
# python eval.py \
#   --model_checkpoint_path ../training/ra_sql_ckpts/ra_sql_sft_model \
#   --tokenizer_name Qwen/Qwen2.5-Coder-3B-Instruct \
#   --task_type ra_sql \
#   --database_path ../data/spider/database \
#   --dataset_name spider \
#   --table_value_cache_path ../data/spider/spider_dev_id2sampled_values.json \
#   --table_info_cache_path ../data/spider/spider_dev_id2db_info.json \
#   --input_file ../data/spider/dev_spider_ra_correct.json \
#   --table_json_path ../data/spider/tables.json \
#   --cot \
#   --tensor_parallel_size 1 \
#   --batch_size 32
