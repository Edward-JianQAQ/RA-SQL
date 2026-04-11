#!/bin/bash
# Section 7.3: Evaluate on extended benchmarks
#   - Spider-dev, Spider-test (core)
#   - Spider-Syn, Spider-Realistic, Spider-DK (robustness variants)
#   - EHRSQL, ScienceBenchmark (specialized domains)
#
# Spider variants reuse Spider's database and schema caches.
# Spider-DK has its own database and caches.
# EHRSQL and ScienceBenchmark require separate database downloads.

cd "$(dirname "$0")/../../evaluation"

MODEL_CHECKPOINT="../training/ra_sql_ckpts/ra_sql_synsql_then_spider"
TOKENIZER="Qwen/Qwen2.5-Coder-7B-Instruct"
TASK_TYPE="ra_sql"

# ---------- Spider-dev ----------
echo "=== Evaluating on Spider-dev ==="
python eval.py \
    --model_checkpoint_path ${MODEL_CHECKPOINT} \
    --tokenizer_name ${TOKENIZER} \
    --database_path ../data/spider/database \
    --dataset_name spider \
    --table_value_cache_path ../data/spider/spider_dev_id2sampled_values.json \
    --table_info_cache_path ../data/spider/spider_dev_id2db_info.json \
    --input_file ../data/spider/dev_spider_ra_correct.json \
    --table_json_path ../data/spider/tables.json \
    --task_type ${TASK_TYPE} \
    --cot \
    --output_log results_spider_dev.json \
    --detailed_log results_spider_dev.txt

# ---------- Spider-test ----------
echo "=== Evaluating on Spider-test ==="
python eval.py \
    --model_checkpoint_path ${MODEL_CHECKPOINT} \
    --tokenizer_name ${TOKENIZER} \
    --database_path ../data/spider/test_database \
    --dataset_name spider \
    --table_value_cache_path ../data/spider/test_id2sampled_values.json \
    --table_info_cache_path ../data/spider/test_id2db_info.json \
    --input_file ../data/spider/test_spider_ra_correct.json \
    --table_json_path ../data/spider/test_tables.json \
    --task_type ${TASK_TYPE} \
    --cot \
    --output_log results_spider_test.json \
    --detailed_log results_spider_test.txt

# ---------- Spider-Syn ----------
echo "=== Evaluating on Spider-Syn ==="
python eval.py \
    --model_checkpoint_path ${MODEL_CHECKPOINT} \
    --tokenizer_name ${TOKENIZER} \
    --database_path ../data/spider/database \
    --dataset_name spider-syn \
    --table_value_cache_path ../data/spider/spider_dev_id2sampled_values.json \
    --table_info_cache_path ../data/spider/spider_dev_id2db_info.json \
    --input_file ../data/spider_syn/dev_spider_syn_ra_correct.json \
    --table_json_path ../data/spider/tables.json \
    --task_type ${TASK_TYPE} \
    --cot \
    --output_log results_spider_syn.json \
    --detailed_log results_spider_syn.txt

# ---------- Spider-Realistic ----------
echo "=== Evaluating on Spider-Realistic ==="
python eval.py \
    --model_checkpoint_path ${MODEL_CHECKPOINT} \
    --tokenizer_name ${TOKENIZER} \
    --database_path ../data/spider/database \
    --dataset_name spider-realistic \
    --table_value_cache_path ../data/spider/spider_dev_id2sampled_values.json \
    --table_info_cache_path ../data/spider/spider_dev_id2db_info.json \
    --input_file ../data/spider_realistic/dev_spider_realistic_ra_correct.json \
    --table_json_path ../data/spider/tables.json \
    --task_type ${TASK_TYPE} \
    --cot \
    --output_log results_spider_realistic.json \
    --detailed_log results_spider_realistic.txt

# ---------- Spider-DK ----------
echo "=== Evaluating on Spider-DK ==="
python eval.py \
    --model_checkpoint_path ${MODEL_CHECKPOINT} \
    --tokenizer_name ${TOKENIZER} \
    --database_path ../data/spider_dk/database \
    --dataset_name spider-dk \
    --table_value_cache_path ../data/spider_dk/spider_dk_id2sampled_values.json \
    --table_info_cache_path ../data/spider_dk/spider_dk_id2db_info.json \
    --input_file ../data/spider_dk/dev_spider_dk_ra_correct.json \
    --table_json_path ../data/spider_dk/tables.json \
    --task_type ${TASK_TYPE} \
    --cot \
    --output_log results_spider_dk.json \
    --detailed_log results_spider_dk.txt

# ---------- EHRSQL ----------
# Requires: EHRSQL database (download from https://physionet.org/)
echo "=== Evaluating on EHRSQL ==="
python eval.py \
    --model_checkpoint_path ${MODEL_CHECKPOINT} \
    --tokenizer_name ${TOKENIZER} \
    --database_path ../data/ehrsql/database \
    --dataset_name ehrsql \
    --table_value_cache_path ../data/ehrsql/id2sampled_values.json \
    --table_info_cache_path ../data/ehrsql/id2db_info.json \
    --input_file ../data/ehrsql/dev_ehrsql_ra_correct.json \
    --table_json_path ../data/ehrsql/tables.json \
    --task_type ${TASK_TYPE} \
    --cot \
    --output_log results_ehrsql.json \
    --detailed_log results_ehrsql.txt

# ---------- ScienceBenchmark ----------
# Requires: ScienceBenchmark databases (cordis, oncomx, sdss_lite)
echo "=== Evaluating on ScienceBenchmark ==="
python eval.py \
    --model_checkpoint_path ${MODEL_CHECKPOINT} \
    --tokenizer_name ${TOKENIZER} \
    --database_path ../data/sciencebenchmark/databases \
    --dataset_name sciencebenchmark \
    --table_value_cache_path ../data/sciencebenchmark/id2sampled_values.json \
    --table_info_cache_path ../data/sciencebenchmark/id2db_info.json \
    --input_file ../data/sciencebenchmark/dev_sciencebenchmark_ra_correct.json \
    --table_json_path ../data/sciencebenchmark/tables.json \
    --task_type ${TASK_TYPE} \
    --cot \
    --output_log results_sciencebenchmark.json \
    --detailed_log results_sciencebenchmark.txt
