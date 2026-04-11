#!/bin/bash
# Section 7.4: Out-of-distribution generalization
#
# Train on Spider → evaluate on BIRD-dev (and vice versa).
# Models are trained with scripts/run_train.sh on one benchmark only,
# then evaluated on the other without any target-domain finetuning.

cd "$(dirname "$0")/../../evaluation"

TASK_TYPE="ra_sql"

# ====================================================================
# Direction 1: Spider-trained model → BIRD-dev
# ====================================================================
echo "=== Spider-trained → BIRD-dev ==="
python eval.py \
    --model_checkpoint_path ../training/ra_sql_ckpts/ra_sql_sft_spider \
    --tokenizer_name Qwen/Qwen2.5-Coder-3B-Instruct \
    --database_path ../data/bird/dev/dev_databases \
    --dataset_name bird \
    --table_value_cache_path ../data/bird/dev/bird_dev_id2sampled_values.json \
    --table_info_cache_path ../data/bird/dev/bird_dev_id2db_info.json \
    --input_file ../data/bird/dev/dev_bird_ra_correct.json \
    --task_type ${TASK_TYPE} \
    --cot \
    --output_log results_ood_spider2bird.json \
    --detailed_log results_ood_spider2bird.txt

# ====================================================================
# Direction 2: BIRD-trained model → Spider-dev
# ====================================================================
echo "=== BIRD-trained → Spider-dev ==="
python eval.py \
    --model_checkpoint_path ../training/ra_sql_ckpts/ra_sql_sft_bird \
    --tokenizer_name Qwen/Qwen2.5-Coder-3B-Instruct \
    --database_path ../data/spider/database \
    --dataset_name spider \
    --table_value_cache_path ../data/spider/spider_dev_id2sampled_values.json \
    --table_info_cache_path ../data/spider/spider_dev_id2db_info.json \
    --input_file ../data/spider/dev_spider_ra_correct.json \
    --table_json_path ../data/spider/tables.json \
    --task_type ${TASK_TYPE} \
    --cot \
    --output_log results_ood_bird2spider.json \
    --detailed_log results_ood_bird2spider.txt
