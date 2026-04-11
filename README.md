# RA-SQL: Relational Algebra as Deterministic Chain-of-Thoughtfor Natural Langauge to SQL


RA-SQL compiles gold SQL queries into relational algebra (RA) operator trees via Apache Calcite and uses the serialized trees as structured chain-of-thought (CoT) supervision for training NL2SQL models. A single autoregressive model generates the RA tree in a `<think>` block, then produces the final SQL in an `<answer>` block.

## Repository Structure

```
RA-SQL/
├── ra_generation/       # SQL-to-RA compilation via Apache Calcite
│   ├── ra_gen.py        # Python driver for batch RA generation
│   ├── model_template.json
│   └── calcite-plan/    # Java Calcite project (Maven)
├── training/            # Supervised fine-tuning
│   ├── train.py         # Main training script
│   ├── data_utils.py    # Data loading and prompt generation
│   ├── training_prompts.py
│   ├── prompt_manager.py
│   ├── serialization/   # RA tree serialization to text
│   └── prompt_templates/
├── evaluation/          # vLLM-based evaluation
│   ├── eval.py          # Main evaluation script
│   ├── eval_spider.py   # SQL evaluation metrics
│   ├── eval_utils.py
│   └── ra_eval_utils.py # RA tree evaluation metrics
└── scripts/             # Example shell scripts
```

## Setup

### Requirements

- Python 3.9+
- Java 11+ and Maven (for building the Calcite compiler)
- CUDA 12.0+ (for training and vLLM inference)

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Build the Calcite compiler

```bash
cd ra_generation/calcite-plan
mvn clean package
cd ../..
```

This produces the JAR file at `ra_generation/calcite-plan/target/calcite-plan-1.0-SNAPSHOT.jar`.

## Data Preparation

### Processed data (included)

The `data/` directory includes pre-processed RA-annotated data files and schema caches for Spider and BIRD, ready for training and evaluation:

```
data/
├── spider/
│   ├── train_spider_ra_correct.json       # 7K training examples with RA
│   ├── dev_spider_ra_correct.json         # 1K dev examples with RA
│   ├── spider_train_id2db_info.json       # Schema cache
│   ├── spider_train_id2sampled_values.json
│   ├── spider_dev_id2db_info.json
│   ├── spider_dev_id2sampled_values.json
│   ├── tables.json                        # Foreign key info
│   └── database/                          # ← download separately
├── bird/
│   ├── train/
│   │   ├── train_bird_ra_correct.json     # 7K training examples with RA
│   │   ├── bird_train_id2db_info.json
│   │   ├── bird_train_id2sampled_values.json
│   │   └── train_databases/               # ← download separately
│   └── dev/
│       ├── dev_bird_ra_correct.json       # 1.3K dev examples with RA
│       ├── bird_dev_id2db_info.json
│       ├── bird_dev_id2sampled_values.json
│       └── dev_databases/                 # ← download separately
```

### Download database files

The SQLite database files are required for training (schema/value retrieval) and evaluation (SQL execution). Download them from the original benchmarks and place under `data/`:

- **Spider**: Download from [Spider](https://yale-lily.github.io/spider) and place the `database/` folder under `data/spider/`
- **BIRD**: Download from [BIRD](https://bird-bench.github.io/) and place `train_databases/` under `data/bird/train/` and `dev_databases/` under `data/bird/dev/`

### (Optional) Regenerate RA trees from scratch

To regenerate the RA-annotated data from raw SQL (requires the Calcite JAR):

```bash
cd ra_generation
python ra_gen.py --datasets spider_train spider_dev --generate-models
python ra_gen.py --datasets bird_train bird_dev --generate-models
```

This produces `*_ra_correct.json` (successfully compiled) and `*_ra_error.json` (compilation failures) for each dataset. Queries that fail to compile are excluded from training.

## Training

Train an RA-SQL model with supervised fine-tuning. The model learns to generate a serialized RA tree as CoT followed by the SQL query.

```bash
cd training

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
  --use_wandb
```

**Task types:**
- `ra_sql` — generate RA tree + SQL (recommended)
- `sql` — generate SQL only
- `ra` — generate RA tree only

**Multi-GPU training** with DeepSpeed ZeRO-3:
```bash
deepspeed --num_gpus 4 train.py \
  --deepspeed --deepspeed_config ds_config_zero3.json \
  ... # same args
```

## Evaluation

Evaluate using vLLM for batch inference:

```bash
cd evaluation

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
  --detailed_log eval_results.txt
```

For Spider evaluation with foreign key support, add `--table_json_path ../data/spider/tables.json`.

**Metrics reported:**
- Execution accuracy (primary metric)
- Exact match accuracy
- Component F1 scores (select, where, group, order, keywords)
- RA tree scores (for `ra` and `ra_sql` tasks)

<details>
<summary><b>Reproducing All Paper Experiments (Section 7)</b></summary>

Scripts for all experiments reported in the paper are in `scripts/experiments/`. Each script includes comments mapping to the corresponding paper section.

### Data requirements

All RA-annotated evaluation data is included in `data/`. You additionally need:

| Data | Required for | How to obtain |
|------|-------------|---------------|
| Spider databases | All Spider evaluations | [Spider](https://yale-lily.github.io/spider) → `data/spider/database/` and `data/spider/test_database/` |
| BIRD databases | BIRD evaluations | [BIRD](https://bird-bench.github.io/) → `data/bird/{train,dev}/` |
| Spider-DK databases | Section 7.3 | [Spider-DK](https://github.com/ygan/Spider-DK) → `data/spider_dk/database/` |
| EHRSQL databases | Section 7.3 | [PhysioNet](https://physionet.org/) (MIMIC-III, eICU) → `data/ehrsql/database/` |
| ScienceBenchmark databases | Section 7.3 | [ScienceBenchmark](https://github.com/anon/sciencebenchmark) → `data/sciencebenchmark/databases/` |
| SynSQL cached dataset | Sections 7.2, 7.5 | Build with `scripts/build_synsql_cache.sh` (see below) |

### Building the SynSQL cached dataset

The SynSQL dataset (2.5M examples) is slow to preprocess at training time because each example requires schema lookup, value retrieval, prompt formatting, RA serialization, and tokenization. We pre-build the dataset once and save it as a pickle file so that subsequent training runs load in seconds:

```bash
# 1. Download SynSQL-2.5M from OmniSQL, run RA generation, create 10% subset
# 2. Build the cache (runs preprocessing once, ~1-2 hours)
bash scripts/build_synsql_cache.sh

# 3. Training loads the cache instantly
python training/train.py --cached_dataset_path data/synsql/cached_datasets/cached_dataset.pkl ...
```

See `scripts/build_synsql_cache.sh` for full instructions on preparing the SynSQL data from scratch.

### Section 7.1 — Main results (RA-SQL vs direct SQL)

Train on Spider or BIRD with `scripts/run_train.sh`, then evaluate with `scripts/run_eval.sh`. Repeat with `--task_type sql` (no RA) for the baseline.

### Section 7.2 — Comparison to state-of-the-art

Two-stage training: SynSQL pretraining followed by benchmark finetuning.

```bash
# Stage 1: Pretrain on SynSQL 10%
bash scripts/experiments/train_synsql_pretrain.sh

# Stage 2: Finetune on Spider (or BIRD)
bash scripts/experiments/train_synsql_then_finetune.sh
```

### Section 7.3 — Extended benchmark evaluation

Evaluate on Spider variants (Spider-Syn, Spider-Realistic, Spider-DK), EHRSQL, and ScienceBenchmark:

```bash
bash scripts/experiments/eval_extended_benchmarks.sh
```

Spider-Syn and Spider-Realistic reuse Spider's database directory and schema caches. Spider-DK, EHRSQL, and ScienceBenchmark require their own database downloads (see table above).

**Evaluation metrics by benchmark:**
- Spider variants: execution accuracy + exact match + component F1 (with foreign key support)
- BIRD: execution accuracy only (BIRD SQL syntax does not support AST-based exact match parsing)
- EHRSQL, ScienceBenchmark: execution accuracy only

### Section 7.4 — Out-of-distribution generalization

Train on one benchmark, evaluate on another without target-domain finetuning:

```bash
bash scripts/experiments/eval_ood.sh
```

### Section 7.5 — CoT ablation (RA-SQL vs LLM-generated CoT vs no CoT)

Train SQL-only baseline (no RA CoT) on the same SynSQL data:

```bash
bash scripts/experiments/train_cot_ablation.sh
```

Then evaluate both variants with the same eval commands, comparing `--task_type ra_sql` (RA-SQL) vs `--task_type sql` (SQL-only).

</details>

## Acknowledgments

- SQL evaluation code adapted from the [Spider benchmark](https://github.com/taoyds/spider) and [OmniSQL](https://github.com/RUCKBReasoning/OmniSQL).
- SQL-to-RA compilation built on [Apache Calcite](https://calcite.apache.org/).
- Benchmarks and Datasets: [Spider](https://yale-lily.github.io/spider), [BIRD](https://bird-bench.github.io/), [SynSQL](https://github.com/RUCKBReasoning/OmniSQL).
