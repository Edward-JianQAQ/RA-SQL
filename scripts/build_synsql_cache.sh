#!/bin/bash
# Build a cached dataset from SynSQL for fast training.
#
# The SynSQL dataset (2.5M examples) takes a long time to preprocess on every
# training run because each example requires database schema lookup, value
# retrieval, prompt formatting, RA serialization, and tokenization.
#
# This script runs the preprocessing once and saves the result as a pickle file.
# Subsequent training runs can load the cache in seconds via --cached_dataset_path.
#
# Prerequisites:
#   1. Download SynSQL-2.5M from OmniSQL (https://github.com/RUCKBReasoning/OmniSQL)
#   2. Run RA generation: cd ra_generation && python ra_gen.py --datasets synsql --generate-models
#   3. Place files under data/synsql/:
#        - synsql_ra_correct_10pct_seed42.json  (10% subset with RA annotations)
#        - synsql_10pct_id2sampled_values.json  (schema value cache for the subset)
#        - synsql_10pct_id2db_info.json         (schema info cache for the subset)
#        - databases/                           (SynSQL SQLite databases)
#
# To create the 10% subset from the full RA-annotated data:
#   python -c "
#   import json, random
#   random.seed(42)
#   data = json.load(open('data/synsql/synsql_ra_correct.json'))
#   subset = random.sample(data, len(data) // 10)
#   json.dump(subset, open('data/synsql/synsql_ra_correct_10pct_seed42.json', 'w'))
#   "

cd "$(dirname "$0")/../training"

# Build cache for RA-SQL task (10% SynSQL)
python prebuild_dataset.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-3B-Instruct \
    --input_file ../data/synsql/synsql_ra_correct_10pct_seed42.json \
    --database_path ../data/synsql/databases \
    --dataset_name synsql \
    --table_value_cache_path ../data/synsql/synsql_10pct_id2sampled_values.json \
    --table_info_cache_path ../data/synsql/synsql_10pct_id2db_info.json \
    --task_type ra_sql \
    --max_length 12800 \
    --cot \
    --serialize_cot_type post_order \
    --apply_template \
    --output_dir ../data/synsql/cached_datasets

echo ""
echo "Done. Use the cache in training with:"
echo "  --cached_dataset_path ../data/synsql/cached_datasets/cached_dataset.pkl"
