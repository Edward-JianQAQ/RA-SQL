#!/usr/bin/env python
"""
Evaluate a trained RA or SQL chat-aligned SFT model on a provided test split.

This version uses chat templates (similar to ra_train_chat.py) for evaluating
Instruct models that were fine-tuned with chat formatting.

Example for RA evaluation:
python ra_eval_chat.py \
  --model_checkpoint_path=ra_sql_ckpts/ra_sft_chat_model \
  --database_path=ra_data/bird/dev_20240627/dev_databases \
  --dataset_name=bird \
  --table_value_cache_path=ra_data/bird/dev_20240627/bird_dev_id2sampled_values.json \
  --table_info_cache_path=ra_data/bird/dev_20240627/bird_dev_id2db_info.json \
  --input_file=ra_data/bird/dev_20240627/dev_bird_ra_correct.json \
  --tokenizer_name=Qwen/Qwen2.5-3B-Instruct \
  --detailed_log evaluation_detailed.txt \
  --verbose 0 \
  --save_errors \
  --mode dev \
  --task_type ra

Example for SQL evaluation (Spider with foreign keys):
python ra_eval_chat.py \
  --model_checkpoint_path=sql_sft_chat_model \
  --database_path=ra_data/spider/database \
  --dataset_name=spider \
  --table_value_cache_path=ra_data/spider/spider_dev_id2sampled_values.json \
  --table_info_cache_path=ra_data/spider/spider_dev_id2db_info.json \
  --input_file=ra_data/bird/dev_20240627/dev_bird_ra_correct.json \
  --table_json_path=ra_data/spider/tables.json \
  --tokenizer_name=Qwen/Qwen2.5-3B-Instruct \
  --detailed_log sql_evaluation_detailed.txt \
  --verbose 0 \
  --save_errors \
  --cot \
  --mode dev \
  --task_type sql

Example for SQL evaluation (BIRD without tables.json):
python ra_eval_chat.py \
  --model_checkpoint_path=sql_sft_chat_model \
  --database_path=ra_data/bird/dev_20240627/dev_databases \
  --dataset_name=bird \
  --table_value_cache_path=ra_data/bird/dev_20240627/bird_dev_id2sampled_values.json \
  --table_info_cache_path=ra_data/bird/dev_20240627/bird_dev_id2db_info.json \
  --input_file=ra_data/bird/dev_20240627/dev_bird_ra_correct.json \
  --tokenizer_name=Qwen/Qwen2.5-3B-Instruct \
  --detailed_log sql_bird_evaluation.txt \
  --verbose 0 \
  --task_type sql

Example for RA+SQL evaluation:
python ra_eval_chat.py \
  --model_checkpoint_path=ra_sql_sft_chat_model \
  --database_path=ra_data/bird/dev_20240627/dev_databases \
  --dataset_name=bird \
  --table_value_cache_path=ra_data/bird/dev_20240627/bird_dev_id2sampled_values.json \
  --table_info_cache_path=ra_data/bird/dev_20240627/bird_dev_id2db_info.json \
  --input_file=ra_data/bird/dev_20240627/dev_bird_ra_correct.json \
  --tokenizer_name=Qwen/Qwen2.5-3B-Instruct \
  --detailed_log ra_sql_evaluation.txt \
  --verbose 0 \
  --task_type ra_sql

Example for CoT evaluation (SynSQL-style):
python ra_eval_chat.py \
  --model_checkpoint_path=synsql_cot_3b_sft \
  --database_path=ra_data/bird/dev_20240627/dev_databases \
  --dataset_name=bird \
  --table_value_cache_path=ra_data/bird/dev_20240627/bird_dev_id2sampled_values.json \
  --table_info_cache_path=ra_data/bird/dev_20240627/bird_dev_id2db_info.json \
  --input_file=ra_data/bird/dev_20240627/dev.json \
  --tokenizer_name=Qwen/Qwen2.5-Coder-3B-Instruct \
  --detailed_log cot_evaluation.txt \
  --verbose 0 \
  --task_type cot
"""

"""
Chat-aligned evaluation script for Relational Algebra, SQL, or combined RA+SQL SFT model using vLLM for fast inference.
Performs detailed evaluation and logs results.

Uses chat template formatting (system + user messages) to match training format.

Supports four task types:
- 'ra': Evaluates Relational Algebra JSON outputs
- 'sql': Evaluates SQL query outputs using Spider evaluation metrics
- 'ra_sql': Evaluates both RA (from thinking) and SQL (from answer) outputs
- 'cot': Evaluates chain-of-thought models (extracts SQL from CoT reasoning for evaluation)

For SQL evaluation, provides:
- Exact match accuracy
- Execution accuracy (if database is accessible)
- Component-wise F1 scores
- Partial matching scores

For RA+SQL evaluation, provides:
- RA exact match accuracy
- SQL exact match accuracy
- Both correct accuracy
- Component scores for both RA and SQL

Example usage for RA:
python ra_eval_chat.py \
    --model_checkpoint_path ./ra_sft_chat_model \
    --database_path /path/to/database \
    --dataset_name spider \
    --table_value_cache_path /path/to/cache \
    --table_info_cache_path /path/to/info_cache \
    --input_file /path/to/test_data.json \
    --output_log evaluation_results.json \
    --task_type ra

Example usage for SQL:
python ra_eval_chat.py \
    --model_checkpoint_path ./sql_sft_chat_model \
    --database_path /path/to/database \
    --dataset_name spider \
    --table_value_cache_path /path/to/cache \
    --table_info_cache_path /path/to/info_cache \
    --input_file /path/to/test_data.json \
    --table_json_path /path/to/tables.json \
    --output_log sql_evaluation_results.json \
    --task_type sql
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from glob import glob
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
import time
import tempfile
from ra_eval_utils import ra_eval_res, parse_ra_output as parse_ra_output_utils
from eval_utils import parse_sql_output, normalize_sql_query
from eval_spider import (
    evaluate_single_pair,
    build_foreign_key_map_from_json
)
from eval_dual_metrics import evaluate_sql_dual_metrics, format_dual_metrics_summary
from eval_bird_style import eval_exec_match_bird_style_simple, eval_exec_match_bird_style_with_results
from eval_spider_official import eval_exec_match_official_simple

# Set environment variables for vLLM to avoid using /tmp
# This should be done before importing vLLM
def configure_vllm_temp_dir(temp_dir_base: Optional[str] = None):
    """Configure vLLM to use a custom temp directory instead of /tmp."""
    if temp_dir_base:
        os.environ['TMPDIR'] = temp_dir_base
        os.environ['TEMP'] = temp_dir_base
        os.environ['TMP'] = temp_dir_base
        # vLLM specific
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        print(f"Configured temp directories to use: {temp_dir_base}")

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Add training directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from data_utils import get_input_seq_ra, get_input_seq, get_input_seq_ra_sql
from training_prompts import (
    extract_json_from_text,
    extract_answer_content,
    extract_sql_from_text,
    extract_thinking_content,
    get_sql_field_from_item
)


DEFAULT_SYSTEM_MESSAGE = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)


class ChatAlignedRelationalAlgebraVLLMEvaluator:
    """Chat-aligned evaluator for Relational Algebra and SQL models using vLLM for fast inference."""

    def __init__(
        self,
        model_checkpoint_path: str,
        tokenizer_name: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 32,
        temp_dir_base: Optional[str] = None,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
    ):
        """Initialize evaluator with vLLM model and generation settings.

        Args:
            model_checkpoint_path: Path to model checkpoint directory
            tokenizer_name: Tokenizer name (if different from checkpoint)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory fraction to use
            max_model_len: Maximum model context length
            max_new_tokens: Maximum new tokens to generate
            temperature: Generation temperature
            top_p: Top-p for nucleus sampling
            batch_size: Batch size for vLLM inference
            temp_dir_base: Base directory for temporary files (default: current directory)
            system_message: System message for chat template
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.model_checkpoint_path = model_checkpoint_path
        self.system_message = system_message

        # Set base directory for temporary files
        if temp_dir_base is None:
            temp_dir_base = os.getcwd()  # Use current directory instead of /tmp

        # Ensure temp base directory exists
        os.makedirs(temp_dir_base, exist_ok=True)

        # Determine tokenizer path
        tokenizer_path = tokenizer_name if tokenizer_name else model_checkpoint_path

        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check if this is a HuggingFace model ID (e.g., "org/model-name") or local path
        is_hf_model_id = not os.path.exists(model_checkpoint_path) and "/" in model_checkpoint_path

        # Check if checkpoint has necessary files (only for local paths)
        checkpoint_files = os.listdir(model_checkpoint_path) if os.path.isdir(model_checkpoint_path) else []
        has_config = "config.json" in checkpoint_files

        # Helper to detect if standard HF weights exist in directory
        def _has_standard_weights(d: str) -> bool:
            if not os.path.isdir(d):
                return False
            # Common HF weight file patterns
            patterns = [
                os.path.join(d, "pytorch_model.bin"),
                os.path.join(d, "model.safetensors"),
                os.path.join(d, "pytorch_model.bin.index.json"),
                os.path.join(d, "model.safetensors.index.json"),
            ]
            for p in patterns:
                if os.path.exists(p):
                    return True
            # Sharded bin/safetensors files
            if glob(os.path.join(d, "pytorch_model-*.bin")):
                return True
            if glob(os.path.join(d, "model-*.safetensors")):
                return True
            return False

        # Prepare model path for vLLM
        model_path_for_vllm = model_checkpoint_path
        temp_dir = None

        # Skip checkpoint file handling for HuggingFace model IDs
        if is_hf_model_id:
            print(f"Detected HuggingFace model ID: {model_checkpoint_path}, will download directly from HF Hub")
            self.temp_dir = None
        elif not has_config and tokenizer_name:
            # Create a temporary directory with config from base model
            print(f"Checkpoint lacks config.json, creating temporary checkpoint with config from {tokenizer_name}...")

            # Create temp directory in specified base directory
            temp_dir = tempfile.mkdtemp(prefix="vllm_checkpoint_", dir=temp_dir_base)
            print(f"Created temporary checkpoint directory: {temp_dir}")

            # Copy checkpoint files
            for file in checkpoint_files:
                src = os.path.join(model_checkpoint_path, file)
                dst = os.path.join(temp_dir, file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

            # Copy config.json from base model
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(repo_id=tokenizer_name, filename="config.json")
            shutil.copy2(config_file, os.path.join(temp_dir, "config.json"))

            # Also copy tokenizer files if not present
            if "tokenizer_config.json" not in checkpoint_files:
                tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "tokenizer.model"]
                for file in tokenizer_files:
                    try:
                        downloaded = hf_hub_download(repo_id=tokenizer_name, filename=file)
                        shutil.copy2(downloaded, os.path.join(temp_dir, file))
                    except:
                        pass  # Some tokenizers don't have all files

            model_path_for_vllm = temp_dir
            self.temp_dir = temp_dir
        else:
            self.temp_dir = None

        # If no standard HF weights are present (typical with DeepSpeed ZeRO-3),
        # attempt to consolidate using zero_to_fp32.py into a temporary folder.
        # Skip this check for HuggingFace model IDs as they will be downloaded directly
        if not is_hf_model_id and not _has_standard_weights(model_checkpoint_path):
            zero_script = None
            # Prefer script in the checkpoint root, else try in sub-checkpoints
            candidate_scripts = [
                os.path.join(model_checkpoint_path, "zero_to_fp32.py"),
            ] + [
                os.path.join(p, "zero_to_fp32.py")
                for p in glob(os.path.join(model_checkpoint_path, "checkpoint-*/"))
            ]
            for c in candidate_scripts:
                if os.path.exists(c):
                    zero_script = c
                    break

            # Determine which folder to pass to zero_to_fp32: use the folder containing a 'latest' file
            # or a folder with global_step* subdir.
            zero_src_dir = None
            latest_file_root = os.path.join(model_checkpoint_path, "latest")
            if os.path.exists(latest_file_root):
                zero_src_dir = model_checkpoint_path
            else:
                # Try within sub checkpoints
                for sub in sorted(glob(os.path.join(model_checkpoint_path, "checkpoint-*/"))):
                    if os.path.exists(os.path.join(sub, "latest")) or glob(os.path.join(sub, "global_step*")):
                        zero_src_dir = sub.rstrip("/")
                        break

            if zero_script and zero_src_dir:
                try:
                    # Create a temporary directory to hold the consolidated HF model
                    temp_dir = tempfile.mkdtemp(prefix="vllm_consolidated_", dir=temp_dir_base)
                    # Copy config/tokenizer files so vLLM finds everything in one place
                    files_to_copy = [
                        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "tokenizer.model", "merges.txt", "vocab.json", "added_tokens.json", "chat_template.jinja"
                    ]
                    for f in files_to_copy:
                        src = os.path.join(model_checkpoint_path, f)
                        if os.path.exists(src) and os.path.isfile(src):
                            shutil.copy2(src, os.path.join(temp_dir, f))

                    # Run consolidation
                    # zero_to_fp32.py expects an output directory, not a single file
                    output_dir = temp_dir
                    print(f"Consolidating DeepSpeed ZeRO-3 checkpoint from {zero_src_dir} into {output_dir} ...")
                    # Use the same python interpreter; patch env to avoid MKL/OMP conflicts seen on some systems
                    cmd = [sys.executable, zero_script, zero_src_dir, output_dir]
                    env = os.environ.copy()
                    # Prefer GNU OpenMP to avoid MKL + libgomp conflicts
                    env["MKL_THREADING_LAYER"] = "GNU"
                    # In case it's needed to bypass mkl-service checks
                    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
                    subprocess.run(cmd, check=True, env=env)
                    print("Consolidation finished successfully.")
                    model_path_for_vllm = temp_dir
                    self.temp_dir = temp_dir
                except Exception as e:
                    # If consolidation fails, raise with guidance
                    msg = (
                        f"Failed to consolidate DeepSpeed ZeRO-3 checkpoint for evaluation: {e}.\n"
                        f"Please ensure DeepSpeed is installed and try manually:\n"
                        f"  python {zero_script or 'zero_to_fp32.py'} {zero_src_dir or model_checkpoint_path} /path/to/output_dir\n"
                        f"Then pass --model_checkpoint_path to the consolidated directory."
                    )
                    raise RuntimeError(msg)

        # Initialize vLLM model
        print(f"Loading model with vLLM from {model_path_for_vllm}...")

        # Set download cache directory if using custom temp dir
        if temp_dir_base:
            download_dir = os.path.join(temp_dir_base, "vllm_downloads")
            os.makedirs(download_dir, exist_ok=True)
            os.environ['HF_HOME'] = download_dir
            os.environ['TRANSFORMERS_CACHE'] = download_dir
            print(f"Using download cache directory: {download_dir}")

        # Ensure MKL/OpenMP configuration is compatible for any subprocesses (e.g., vLLM inspectors)
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

        try:
            vllm_kwargs = {
                "model": model_path_for_vllm,
                "tokenizer": tokenizer_path,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "download_dir": download_dir if temp_dir_base else None,
            }

            if max_model_len:
                vllm_kwargs["max_model_len"] = max_model_len

            self.model = LLM(**vllm_kwargs)
            print("vLLM model loaded successfully!")

        except Exception as e:
            print(f"Error loading model with vLLM: {e}")
            if self.temp_dir:
                shutil.rmtree(self.temp_dir)
            raise

    def __del__(self):
        """Clean up temporary directory if created."""
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temp directory {self.temp_dir}: {e}")

    def parse_ra_output(self, text: str) -> Optional[Dict]:
        """Parse relational algebra JSON from model output.

        Uses the centralized extraction functions for consistency.
        First tries to extract from answer section, then from full text.
        """
        return parse_ra_output_utils(
            text,
            extract_answer_func=extract_answer_content,
            extract_json_func=extract_json_from_text
        )

    def parse_sql_output(self, text: str) -> Optional[str]:
        """Parse SQL query from model output.

        Uses the centralized extraction functions for consistency.
        First tries to extract from answer section, then from full text.
        """
        return parse_sql_output(
            text,
            extract_answer_func=extract_answer_content,
            extract_sql_func=extract_sql_from_text
        )

    def format_chat_prompt(self, prompt_text: str) -> str:
        """Format prompt with chat template.

        Args:
            prompt_text: The raw prompt text

        Returns:
            Chat-formatted prompt string
        """
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt_text})

        # Apply chat template with generation prompt
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return chat_prompt

    def generate_batch_predictions(self, prompts: List[str]) -> List[tuple[str, float]]:
        """Generate predictions for a batch of prompts using vLLM.

        Args:
            prompts: List of raw prompt texts (will be formatted with chat template)

        Returns:
            List of (generated_text, generation_time) tuples
        """
        start_time = time.time()

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            stop=[self.tokenizer.eos_token] if self.tokenizer.eos_token else None,
        )

        # Format prompts with chat template
        formatted_prompts = [self.format_chat_prompt(p) for p in prompts]

        # Generate with vLLM
        outputs = self.model.generate(formatted_prompts, sampling_params)

        # Extract generated texts
        results = []
        gen_time_per_sample = (time.time() - start_time) / len(prompts)

        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            results.append((generated_text, gen_time_per_sample))

        return results

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        database_path: str,
        dataset_name: str,
        table_value_cache_path: str,
        table_info_cache_path: str,
        model_checkpoint_path: str,
        tokenizer_name: Optional[str] = None,
        max_samples: Optional[int] = None,
        verbose: int = 0,
        detailed_log_path: Optional[str] = None,
        cot: bool = False,
        mode: str = 'dev',
        debug: bool = False,
        log_ra_pair_path: Optional[str] = None,
        task_type: str = 'ra',
        table_json_path: Optional[str] = None,
        save_exec_details: bool = False,
        exec_timeout: Optional[int] = None,
        skip_ra_eval: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate model on dataset with detailed logging using vLLM batching.

        Args:
            dataset: List of evaluation samples
            database_path: Path to database
            dataset_name: Name of dataset
            table_value_cache_path: Path to table value cache
            table_info_cache_path: Path to table info cache
            model_checkpoint_path: Path to model checkpoint
            tokenizer_name: Optional tokenizer name
            max_samples: Maximum number of samples to evaluate
            verbose: Verbosity level (0=summary, 1=first 10 + errors, 2=all)
            detailed_log_path: Optional path to save detailed text log
            cot: Chain of thought reasoning
            mode: Evaluation mode (train/dev)
            debug: Debug mode
            log_ra_pair_path: Path to save RA pairs
            task_type: Task type ('ra', 'sql', or 'ra_sql')
            table_json_path: Path to tables.json for SQL evaluation
            save_exec_details: Save execution result details
            exec_timeout: SQL execution timeout in seconds (None = no timeout)

        Returns:
            Dictionary containing evaluation results with metrics and detailed outputs
        """

        # Load foreign key maps if evaluating SQL, RA+SQL, or CoT (optional, mainly for Spider)
        kmaps = None
        if task_type in ['sql', 'ra_sql', 'cot'] and table_json_path and os.path.exists(table_json_path):
            print(f"Loading foreign key maps from {table_json_path}...")
            # import pdb; pdb.set_trace()
            try:
                kmaps = build_foreign_key_map_from_json(table_json_path)
            except Exception as e:
                print(f"Warning: Could not load foreign key maps: {e}")
                print("Continuing without foreign key maps (may affect accuracy slightly)")
                kmaps = None

        # Load Spider2 evaluation standard if provided

        results = {
            "metadata": {
                "model_checkpoint": model_checkpoint_path,
                "tokenizer": tokenizer_name or model_checkpoint_path,
                "evaluation_timestamp": datetime.now().isoformat(),
                "num_samples": min(len(dataset), max_samples) if max_samples else len(dataset),
                "task_type": task_type,
                "chat_aligned": True,
                "system_message": self.system_message,
                "generation_config": {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "batch_size": self.batch_size
                },
                "inference_engine": "vLLM"
            },
            "overall_metrics": {
                "total": 0,
                "correct": 0,
                "parse_errors": 0,
                "accuracy": 0.0,
                "avg_generation_time": 0.0,
                "total_generation_time": 0.0,
                "score": 0.0,
                "component_recall_score": 0.0
            },
            "detailed_results": []
        }

        # Add SQL-specific metrics if evaluating SQL, RA+SQL, or CoT
        if task_type in ['sql', 'ra_sql', 'cot']:
            results["overall_metrics"]["exec_accuracy"] = 0.0
            results["overall_metrics"]["exec_correct"] = 0
            # BIRD-style metrics (official benchmark)
            results["overall_metrics"]["exec_bird_accuracy"] = 0.0
            results["overall_metrics"]["exec_bird_correct"] = 0
            # Official Spider metrics (official benchmark, more lenient than res_map)
            results["overall_metrics"]["exec_spider_official_accuracy"] = 0.0
            results["overall_metrics"]["exec_spider_official_correct"] = 0
            results["overall_metrics"]["exec_methods_agree"] = 0
            results["overall_metrics"]["exec_bird_more_lenient"] = 0

        # Add RA+SQL specific metrics
        if task_type == 'ra_sql':
            results["overall_metrics"]["ra_correct"] = 0
            results["overall_metrics"]["sql_correct"] = 0
            results["overall_metrics"]["both_correct"] = 0
            results["overall_metrics"]["ra_accuracy"] = 0.0
            results["overall_metrics"]["sql_accuracy"] = 0.0
            results["overall_metrics"]["both_accuracy"] = 0.0
            results["overall_metrics"]["ra_score"] = 0.0
            results["overall_metrics"]["sql_score"] = 0.0
            results["overall_metrics"]["ra_component_recall_score"] = 0.0
            results["overall_metrics"]["sql_component_recall_score"] = 0.0

        eval_samples = dataset[:max_samples] if max_samples else dataset

        # Process in batches
        batch_prompts = []
        batch_items = []
        batch_indices = []

        print("Preparing prompts...")
        for idx, item in enumerate(tqdm(eval_samples, desc="Preparing")):
            # Get prompt based on task type
            if task_type == 'ra':
                prompt = get_input_seq_ra(
                    item,
                    database_path,
                    dataset_name,
                    table_value_cache_path,
                    table_info_cache_path,
                    mode=mode,
                    cot=cot
                )
            elif task_type == 'ra_sql':
                prompt = get_input_seq_ra_sql(
                    item,
                    database_path,
                    dataset_name,
                    table_value_cache_path,
                    table_info_cache_path,
                    mode=mode,
                    cot=cot
                )
            elif task_type == 'cot':
                # CoT task: use SQL-style prompt (question + schema)
                # Model generates chain-of-thought reasoning that includes SQL
                prompt = get_input_seq(
                    item,
                    database_path,
                    dataset_name,
                    table_value_cache_path,
                    table_info_cache_path,
                    mode=mode,
                    cot=False  # Don't add CoT in prompt since model generates it
                )
            else:  # SQL task
                prompt = get_input_seq(
                    item,
                    database_path,
                    dataset_name,
                    table_value_cache_path,
                    table_info_cache_path,
                    mode=mode,
                    cot=cot
                )

            batch_prompts.append(prompt)
            batch_items.append(item)
            batch_indices.append(idx)

        # Generate predictions in batches
        all_predictions = []
        print(f"Generating predictions in batches of {self.batch_size}...")

        for i in tqdm(range(0, len(batch_prompts), self.batch_size), desc="Batch inference"):
            batch_end = min(i + self.batch_size, len(batch_prompts))
            batch = batch_prompts[i:batch_end]

            # Generate batch predictions
            batch_results = self.generate_batch_predictions(batch)
            all_predictions.extend(batch_results)

        # Process results
        print("Processing results...")
        total_gen_time = 0.0

        # Open detailed log file if specified
        detailed_log_file = None
        if detailed_log_path:
            detailed_log_file = open(detailed_log_path, 'w', encoding='utf-8')
            detailed_log_file.write(f"Evaluation Log (Chat-Aligned) - {datetime.now().isoformat()}\n")
            detailed_log_file.write(f"Model: {model_checkpoint_path}\n")
            detailed_log_file.write(f"Tokenizer: {tokenizer_name or model_checkpoint_path}\n")
            detailed_log_file.write(f"Dataset: {dataset_name}\n")
            detailed_log_file.write(f"Chat Template: Enabled\n")
            detailed_log_file.write(f"System Message: {self.system_message}\n")
            detailed_log_file.write(f"Total Samples to Evaluate: {len(eval_samples)}\n")
            detailed_log_file.write(f"{'='*100}\n\n")

        if log_ra_pair_path and debug:
            log_json = []

        for idx, (item, prompt, (pred_text, gen_time)) in enumerate(
            tqdm(zip(batch_items, batch_prompts, all_predictions), desc="Processing", total=len(batch_items))
        ):
            results["overall_metrics"]["total"] += 1
            total_gen_time += gen_time

            # if idx in [361, 368, 417, 426, 427,428] and dataset_name == "bird":  # For debugging specific samples
            #     print("skipping sample 361")
            #     continue
            # if idx in [395, 396, 426 , 427, 428] and dataset_name == "bird":  # For debugging specific samples
            #     print("skipping sample 395, 396, 426, 427, 428")
            #     continue
            # if idx in [732, 733, 734, 735] and dataset_name == "spider":  # For debugging specific samples
            #     # import pdb; pdb.set_trace()
            # Initialize common variables
            is_correct = False
            parse_error = False
            error_message = None
            cont_res = {
                'score': 0,
                'component_recall_score': 0
            }
            exec_accuracy = None  # For SQL evaluation (Spider-style res_map)
            exec_bird = None      # For SQL evaluation (BIRD-style)
            exec_spider_official = None  # For SQL evaluation (Official Spider)
            pred_exec_results = None  # Actual query results from predicted SQL
            gold_exec_results = None  # Actual query results from gold SQL

            if task_type == 'ra':
                # Get ground truth RA
                gold_ra = item["relational_algebra"]
                gold_ra_str = json.dumps(gold_ra, indent=2)

                # Parse prediction
                pred_ra = self.parse_ra_output(pred_text)

                if pred_ra is None:
                    parse_error = True
                    error_message = "Failed to parse JSON from model output"
                    results["overall_metrics"]["parse_errors"] += 1
                else:
                    # Compute continuous scores
                    try:
                        cont_res = ra_eval_res(gold_ra, pred_ra, debug=debug)
                    except:
                        # When pred text fails to form a valid ra
                        cont_res = {
                            'score': 0,
                            'component_recall_score': 0
                        }
                    is_correct = (pred_ra == gold_ra)
                    if is_correct:
                        results["overall_metrics"]["correct"] += 1

                results['overall_metrics']['score'] += cont_res['score']
                results['overall_metrics']['component_recall_score'] += cont_res['component_recall_score']

                if log_ra_pair_path and debug:
                    # save as a json dict
                    log_entry = {
                        "question_id": item.get("question_id", f"sample_{idx}"),
                        "question": item.get("question", ""),
                        "SQL": item.get("SQL", ""),
                        "ground_truth": gold_ra,
                        "model_output_raw": pred_text,
                        "model_output_parsed": pred_ra,
                        "is_correct": is_correct,
                        "parse_error": parse_error,
                        "error_message": error_message,
                        "continuous_result": cont_res
                    }
                    log_json.append(log_entry)

            elif task_type == 'ra_sql':
                # RA+SQL evaluation - evaluate both components (or SQL only if skip_ra_eval=True)

                # Get ground truth
                gold_ra = item.get("relational_algebra", None)  # May not exist if skip_ra_eval=True
                gold_sql = get_sql_field_from_item(item, dataset_name)
                db_id = item.get("db_id", "")

                # Initialize tracking variables
                ra_match = False
                sql_match = False
                ra_cont_res = {'score': 0, 'component_recall_score': 0}
                sql_cont_res = {'score': 0, 'component_recall_score': 0}
                pred_ra = None

                # Extract and evaluate RA from thinking section (skip if skip_ra_eval=True or no RA ground truth)
                if not skip_ra_eval and gold_ra is not None:
                    thinking_content = extract_thinking_content(pred_text)
                    if thinking_content:
                        pred_ra = self.parse_ra_output(thinking_content)
                    else:
                        # Try extracting from whole text if no thinking section
                        pred_ra = self.parse_ra_output(pred_text)

                    if pred_ra is None:
                        parse_error = True
                        error_message = "Failed to parse RA JSON from model output"
                        results["overall_metrics"]["parse_errors"] += 1
                    else:
                        # Compute RA continuous scores
                        try:
                            ra_cont_res = ra_eval_res(gold_ra, pred_ra, debug=debug)
                        except:
                            ra_cont_res = {'score': 0, 'component_recall_score': 0}

                        ra_match = (pred_ra == gold_ra)
                        if ra_match:
                            results["overall_metrics"]["ra_correct"] += 1

                    results['overall_metrics']['ra_score'] += ra_cont_res['score']
                    results['overall_metrics']['ra_component_recall_score'] += ra_cont_res['component_recall_score']

                # Extract and evaluate SQL
                pred_sql = self.parse_sql_output(pred_text)

                if pred_sql is None:
                    if not parse_error:  # Only set if RA parsing didn't already fail
                        parse_error = True
                        error_message = "Failed to parse SQL from model output"
                        results["overall_metrics"]["parse_errors"] += 1
                # Spider2-specific evaluation: compare against gold execution results (CSV files)

                else:
                    # Evaluate SQL using Spider evaluation
                    try:
                        if kmaps is not None:
                            eval_result = evaluate_sql_dual_metrics(
                                pred_sql=pred_sql,
                                gold_sql=gold_sql,
                                db_dir=database_path,
                                db_name=db_id,
                                etype='all',
                                kmaps=kmaps,
                                dataset_name=dataset_name,  # Pass dataset name for consistent exact match
                                exec_timeout=exec_timeout
                            )
                        else:
                            # Create empty kmaps dict for evaluation
                            empty_kmaps = {db_id: {}} if db_id else {}
                            eval_result = evaluate_sql_dual_metrics(
                                pred_sql=pred_sql,
                                gold_sql=gold_sql,
                                db_dir=database_path,
                                db_name=db_id,
                                etype='all',
                                kmaps=empty_kmaps,
                                dataset_name=dataset_name,  # Pass dataset name for consistent exact match
                                exec_timeout=exec_timeout
                            )

                        # Extract SQL scores
                        sql_match = eval_result['exact'] == 1
                        exec_accuracy = eval_result['exec_spider']  # Spider-style
                        exec_bird = eval_result['exec_bird']        # BIRD-style

                        # Try official Spider execution (most lenient method)
                        # Only run for Spider dataset variants (slow operation, not needed for BIRD-style)
                        if dataset_name in ['spider', 'spider-dk', 'spider-syn', 'spider-realistic', 'spider-dk-omnisql']:
                            try:
                                db_path = os.path.join(database_path, db_id, f"{db_id}.sqlite")
                                if os.path.exists(db_path):
                                    official_success = eval_exec_match_official_simple(db_path, pred_sql, gold_sql)
                                    exec_spider_official = 1 if official_success else 0
                                    if exec_spider_official == 1:
                                        results["overall_metrics"]["exec_spider_official_correct"] += 1
                                    if verbose > 1 or debug:
                                        print(f"[DEBUG] Official Spider execution: {official_success}")
                                else:
                                    exec_spider_official = None
                            except Exception as official_e:
                                exec_spider_official = None
                                if verbose > 1 or debug:
                                    print(f"[DEBUG] Official Spider execution failed: {official_e}")
                        else:
                            # Skip official Spider execution for non-Spider datasets (e.g., BIRD, ehrsql, sciencebenchmark)
                            exec_spider_official = None

                        if verbose > 1 or debug:
                            print(f"[DEBUG] Sample {item.get('question_id')}: exec_spider={exec_accuracy}, exec_bird={exec_bird}, exec_spider_official={exec_spider_official}")

                        if eval_result['partial']:
                            partial_scores = eval_result['partial']
                            f1_scores = [v['f1'] for v in partial_scores.values()]
                            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                            sql_cont_res = {
                                'score': avg_f1,
                                'component_recall_score': avg_f1,
                                'partial_scores': partial_scores
                            }
                        else:
                            sql_cont_res = {
                                'score': 1.0 if sql_match else 0.0,
                                'component_recall_score': 1.0 if sql_match else 0.0
                            }

                        if sql_match:
                            results["overall_metrics"]["sql_correct"] += 1
                        if exec_accuracy == 1:
                            results["overall_metrics"]["exec_correct"] += 1

                        # Track BIRD-style execution
                        if exec_bird == 1:
                            results["overall_metrics"]["exec_bird_correct"] += 1
                        if eval_result.get('exec_methods_agree'):
                            results["overall_metrics"]["exec_methods_agree"] += 1
                        if eval_result.get('exec_bird_more_lenient'):
                            results["overall_metrics"]["exec_bird_more_lenient"] += 1
                    except Exception as e:
                        # Fallback to simple string comparison + BIRD execution
                        if debug or verbose > 0:
                            print(f"Warning: Full SQL evaluation failed, using string comparison + BIRD execution: {e}")
                            if verbose > 1:
                                import traceback
                                traceback.print_exc()

                        # String comparison for SQL match
                        gold_normalized = normalize_sql_query(gold_sql)
                        pred_normalized = normalize_sql_query(pred_sql)
                        sql_match = (gold_normalized == pred_normalized)
                        if sql_match:
                            results["overall_metrics"]["sql_correct"] += 1
                        sql_cont_res = {
                            'score': 1.0 if sql_match else 0.0,
                            'component_recall_score': 1.0 if sql_match else 0.0
                        }

                        # Try BIRD execution even if Spider parsing failed
                        exec_accuracy = None  # Spider parsing failed, can't get Spider exec
                        try:
                            db_path = os.path.join(database_path, db_id, f"{db_id}.sqlite")
                            if os.path.exists(db_path):
                                bird_success, bird_error, pred_exec_results, gold_exec_results = eval_exec_match_bird_style_with_results(db_path, pred_sql, gold_sql)
                                exec_bird = 1 if bird_success else 0
                                if exec_bird == 1:
                                    results["overall_metrics"]["exec_bird_correct"] += 1
                                if verbose > 1 or debug:
                                    print(f"[DEBUG] BIRD execution fallback: {bird_success}")
                                    if not bird_success and bird_error:
                                        print(f"[DEBUG] BIRD error: {bird_error}")
                            else:
                                exec_bird = None
                                if verbose > 1 or debug:
                                    print(f"[DEBUG] Database not found at {db_path}")
                        except Exception as bird_e:
                            exec_bird = None
                            if verbose > 1 or debug:
                                print(f"[DEBUG] BIRD execution also failed: {bird_e}")

                        # Try official Spider execution (more lenient than res_map)
                        # Only run for Spider dataset variants (slow operation, not needed for BIRD-style)
                        if dataset_name in ['spider', 'spider-dk', 'spider-syn', 'spider-realistic', 'spider-dk-omnisql']:
                            try:
                                db_path = os.path.join(database_path, db_id, f"{db_id}.sqlite")
                                if os.path.exists(db_path):
                                    official_success = eval_exec_match_official_simple(db_path, pred_sql, gold_sql)
                                    exec_spider_official = 1 if official_success else 0
                                    if exec_spider_official == 1:
                                        results["overall_metrics"]["exec_spider_official_correct"] += 1
                                    if verbose > 1 or debug:
                                        print(f"[DEBUG] Official Spider execution: {official_success}")
                                else:
                                    exec_spider_official = None
                            except Exception as official_e:
                                exec_spider_official = None
                                if verbose > 1 or debug:
                                    print(f"[DEBUG] Official Spider execution failed: {official_e}")
                        else:
                            # Skip official Spider execution for non-Spider datasets (e.g., BIRD)
                            exec_spider_official = None

                results['overall_metrics']['sql_score'] += sql_cont_res['score']
                results['overall_metrics']['sql_component_recall_score'] += sql_cont_res['component_recall_score']

                # Track both correct
                if ra_match and sql_match:
                    results["overall_metrics"]["both_correct"] += 1
                    results["overall_metrics"]["correct"] += 1
                    is_correct = True
                else:
                    is_correct = False

                # Combine continuous results for overall score
                cont_res = {
                    'score': (ra_cont_res['score'] + sql_cont_res['score']) / 2,
                    'component_recall_score': (ra_cont_res['component_recall_score'] + sql_cont_res['component_recall_score']) / 2,
                    'ra_score': ra_cont_res['score'],
                    'sql_score': sql_cont_res['score'],
                    'ra_component_recall_score': ra_cont_res['component_recall_score'],
                    'sql_component_recall_score': sql_cont_res['component_recall_score']
                }

                results['overall_metrics']['score'] += cont_res['score']
                results['overall_metrics']['component_recall_score'] += cont_res['component_recall_score']

            else:  # SQL or CoT evaluation
                # For SQL: model outputs SQL directly
                # For CoT: model outputs chain-of-thought reasoning that includes SQL
                # In both cases, we extract and evaluate the SQL query

                # Get ground truth SQL using dataset-aware function
                # Spider variants use 'query' field, BIRD/OmniSQL uses 'SQL' or 'sql' field
                if dataset_name.lower() in ['spider', 'spider-dk', 'spider-syn', 'spider-realistic', 'spider-dk-omnisql']:
                    gold_sql = item.get("query", item.get("SQL", ""))
                else:
                    # BIRD, ehrsql, sciencebenchmark use 'SQL' or 'sql'
                    gold_sql = item.get("SQL", item.get("sql", ""))
                db_id = item.get("db_id", "")

                # Initialize execution variables
                exec_accuracy = None
                exec_bird = None

                # Parse predicted SQL
                pred_sql = self.parse_sql_output(pred_text)

                if pred_sql is None:
                    parse_error = True
                    error_message = "Failed to parse SQL from model output"
                    results["overall_metrics"]["parse_errors"] += 1
                    cont_res = {'score': 0, 'component_recall_score': 0}
                # Spider2-specific evaluation: compare against gold execution results (CSV files)

                else:
                    # Evaluate SQL using Spider evaluation
                    try:
                        if kmaps is not None:
                            eval_result = evaluate_sql_dual_metrics(
                                pred_sql=pred_sql,
                                gold_sql=gold_sql,
                                db_dir=database_path,
                                db_name=db_id,
                                etype='all',  # Evaluate both execution and matching
                                kmaps=kmaps,
                                dataset_name=dataset_name  # Pass dataset name for consistent exact match
                            )

                            # Extract scores
                            is_correct = eval_result['exact'] == 1
                            exec_accuracy = eval_result['exec_spider']  # Spider-style
                            exec_bird = eval_result['exec_bird']        # BIRD-style

                            # Map partial scores to continuous scores
                            if eval_result['partial']:
                                # Calculate average F1 score across components
                                partial_scores = eval_result['partial']
                                f1_scores = [v['f1'] for v in partial_scores.values()]
                                avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                                cont_res = {
                                    'score': avg_f1,
                                    'component_recall_score': avg_f1,
                                    'partial_scores': partial_scores
                                }
                            else:
                                cont_res = {
                                    'score': 1.0 if is_correct else 0.0,
                                    'component_recall_score': 1.0 if is_correct else 0.0
                                }

                            if is_correct:
                                results["overall_metrics"]["correct"] += 1
                            if exec_accuracy == 1:
                                results["overall_metrics"]["exec_correct"] += 1
                            # Track BIRD execution
                            if exec_bird == 1:
                                results["overall_metrics"]["exec_bird_correct"] += 1

                            # Try official Spider execution (most lenient method)
                            # Only run for Spider dataset variants (slow operation, not needed for BIRD-style)
                            if dataset_name in ['spider', 'spider-dk', 'spider-syn', 'spider-realistic', 'spider-dk-omnisql']:
                                try:
                                    db_path = os.path.join(database_path, db_id, f"{db_id}.sqlite")
                                    if os.path.exists(db_path):
                                        official_success = eval_exec_match_official_simple(db_path, pred_sql, gold_sql)
                                        exec_spider_official = 1 if official_success else 0
                                        if exec_spider_official == 1:
                                            results["overall_metrics"]["exec_spider_official_correct"] += 1
                                        if verbose > 1 or debug:
                                            print(f"[DEBUG] Official Spider execution: {official_success}")
                                    else:
                                        exec_spider_official = None
                                except Exception as official_e:
                                    exec_spider_official = None
                                    if verbose > 1 or debug:
                                        print(f"[DEBUG] Official Spider execution failed: {official_e}")
                            else:
                                # Skip official Spider execution for non-Spider datasets (e.g., BIRD)
                                exec_spider_official = None
                        else:
                            # Evaluate without foreign key maps (works for BIRD or when tables.json unavailable)
                            try:
                                # Create empty kmaps dict for evaluation
                                empty_kmaps = {db_id: {}} if db_id else {}
                                eval_result = evaluate_sql_dual_metrics(
                                    pred_sql=pred_sql,
                                    gold_sql=gold_sql,
                                    db_dir=database_path,
                                    db_name=db_id,
                                    etype='all',
                                    kmaps=empty_kmaps,
                                    dataset_name=dataset_name  # Pass dataset name for consistent exact match
                                )

                                # Extract scores
                                is_correct = eval_result['exact'] == 1
                                exec_accuracy = eval_result['exec_spider']  # Spider-style
                                exec_bird = eval_result['exec_bird']        # BIRD-style

                                # Map partial scores to continuous scores
                                if eval_result['partial']:
                                    partial_scores = eval_result['partial']
                                    f1_scores = [v['f1'] for v in partial_scores.values()]
                                    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                                    cont_res = {
                                        'score': avg_f1,
                                        'component_recall_score': avg_f1,
                                        'partial_scores': partial_scores
                                    }
                                else:
                                    cont_res = {
                                        'score': 1.0 if is_correct else 0.0,
                                        'component_recall_score': 1.0 if is_correct else 0.0
                                    }

                                if is_correct:
                                    results["overall_metrics"]["correct"] += 1
                                if exec_accuracy == 1:
                                    results["overall_metrics"]["exec_correct"] += 1
                                # Track BIRD execution
                                if exec_bird == 1:
                                    results["overall_metrics"]["exec_bird_correct"] += 1

                                # Try official Spider execution (most lenient method)
                                try:
                                    db_path = os.path.join(database_path, db_id, f"{db_id}.sqlite")
                                    if os.path.exists(db_path):
                                        official_success = eval_exec_match_official_simple(db_path, pred_sql, gold_sql)
                                        exec_spider_official = 1 if official_success else 0
                                        if exec_spider_official == 1:
                                            results["overall_metrics"]["exec_spider_official_correct"] += 1
                                        if verbose > 1 or debug:
                                            print(f"[DEBUG] Official Spider execution: {official_success}")
                                    else:
                                        exec_spider_official = None
                                except Exception as official_e:
                                    exec_spider_official = None
                                    if verbose > 1 or debug:
                                        print(f"[DEBUG] Official Spider execution failed: {official_e}")

                            except Exception as e:
                                # Fallback to simple string comparison + BIRD execution
                                if debug or verbose > 0:
                                    print(f"Warning: Full SQL evaluation failed, using string comparison + BIRD execution: {e}")
                                    if verbose > 1:
                                        import traceback
                                        traceback.print_exc()

                                gold_normalized = normalize_sql_query(gold_sql)
                                pred_normalized = normalize_sql_query(pred_sql)
                                is_correct = (gold_normalized == pred_normalized)
                                if is_correct:
                                    results["overall_metrics"]["correct"] += 1
                                cont_res = {
                                    'score': 1.0 if is_correct else 0.0,
                                    'component_recall_score': 1.0 if is_correct else 0.0
                                }

                                # Try BIRD execution even if Spider parsing failed
                                exec_accuracy = None
                                try:
                                    db_path = os.path.join(database_path, db_id, f"{db_id}.sqlite")
                                    if os.path.exists(db_path):
                                        bird_success, bird_error, pred_exec_results, gold_exec_results = eval_exec_match_bird_style_with_results(db_path, pred_sql, gold_sql)
                                        exec_bird = 1 if bird_success else 0
                                        if exec_bird == 1:
                                            results["overall_metrics"]["exec_bird_correct"] += 1
                                        if verbose > 1 or debug:
                                            print(f"[DEBUG] BIRD execution fallback: {bird_success}")
                                            if not bird_success and bird_error:
                                                print(f"[DEBUG] BIRD error: {bird_error}")
                                    else:
                                        exec_bird = None
                                except Exception as bird_e:
                                    exec_bird = None
                                    if verbose > 1 or debug:
                                        print(f"[DEBUG] BIRD execution also failed: {bird_e}")

                                # Try official Spider execution (more lenient than res_map)
                                try:
                                    db_path = os.path.join(database_path, db_id, f"{db_id}.sqlite")
                                    if os.path.exists(db_path):
                                        official_success = eval_exec_match_official_simple(db_path, pred_sql, gold_sql)
                                        exec_spider_official = 1 if official_success else 0
                                        if exec_spider_official == 1:
                                            results["overall_metrics"]["exec_spider_official_correct"] += 1
                                        if verbose > 1 or debug:
                                            print(f"[DEBUG] Official Spider execution: {official_success}")
                                    else:
                                        exec_spider_official = None
                                except Exception as official_e:
                                    exec_spider_official = None
                                    if verbose > 1 or debug:
                                        print(f"[DEBUG] Official Spider execution failed: {official_e}")
                    except Exception as e:
                        error_message = f"SQL evaluation error: {str(e)}"
                        cont_res = {'score': 0, 'component_recall_score': 0}
                        exec_accuracy = None
                        exec_bird = None
                        exec_spider_official = None
                        if debug:
                            print(f"Error evaluating SQL: {e}")

                results['overall_metrics']['score'] += cont_res['score']
                results['overall_metrics']['component_recall_score'] += cont_res['component_recall_score']

            # Create detailed result entry
            result_entry = {
                "index": batch_indices[idx],
                "question_id": item.get("question_id", f"sample_{idx}"),
                "question": item.get("question", ""),
                "prompt": prompt,
                "is_correct": is_correct,
                "parse_error": parse_error,
                "error_message": error_message,
                "generation_time_seconds": gen_time,
                "cont_res": cont_res
            }

            if task_type == 'ra':
                result_entry["ground_truth"] = gold_ra
                result_entry["ground_truth_str"] = gold_ra_str
                result_entry["model_output_parsed"] = pred_ra
            elif task_type == 'ra_sql':
                result_entry["ground_truth_ra"] = gold_ra
                result_entry["ground_truth_sql"] = gold_sql
                result_entry["predicted_ra"] = pred_ra
                result_entry["predicted_sql"] = pred_sql
                result_entry["ra_match"] = ra_match
                result_entry["sql_match"] = sql_match
                if exec_accuracy is not None:
                    result_entry["exec_accuracy"] = exec_accuracy
                # Save detailed execution results if flag is enabled
                if save_exec_details:
                    if verbose > 1 or debug:
                        print(f"[DEBUG] RA_SQL: Saving exec details for sample {item.get('question_id')}: exec_spider={exec_accuracy}, exec_bird={exec_bird}, exec_spider_official={exec_spider_official}")
                    if exec_accuracy is not None:
                        result_entry["exec_spider"] = exec_accuracy
                        result_entry["exec_spider_result"] = "yes" if exec_accuracy == 1 else "no"
                    if exec_bird is not None:
                        result_entry["exec_bird"] = exec_bird
                        result_entry["exec_bird_result"] = "yes" if exec_bird == 1 else "no"
                    if exec_spider_official is not None:
                        result_entry["exec_spider_official"] = exec_spider_official
                        result_entry["exec_spider_official_result"] = "yes" if exec_spider_official == 1 else "no"
                    # Save actual execution results
                    if pred_exec_results is not None:
                        result_entry["predicted_execution_results"] = pred_exec_results
                    if gold_exec_results is not None:
                        result_entry["gold_execution_results"] = gold_exec_results
            else:  # SQL
                result_entry["ground_truth_sql"] = gold_sql
                result_entry["predicted_sql"] = pred_sql
                if exec_accuracy is not None:
                    result_entry["exec_accuracy"] = exec_accuracy
                # Save detailed execution results if flag is enabled
                if save_exec_details:
                    if exec_accuracy is not None:
                        result_entry["exec_spider"] = exec_accuracy
                        result_entry["exec_spider_result"] = "yes" if exec_accuracy == 1 else "no"
                    if exec_bird is not None:
                        result_entry["exec_bird"] = exec_bird
                        result_entry["exec_bird_result"] = "yes" if exec_bird == 1 else "no"
                    if exec_spider_official is not None:
                        result_entry["exec_spider_official"] = exec_spider_official
                        result_entry["exec_spider_official_result"] = "yes" if exec_spider_official == 1 else "no"
                    # Save actual execution results
                    if pred_exec_results is not None:
                        result_entry["predicted_execution_results"] = pred_exec_results
                    if gold_exec_results is not None:
                        result_entry["gold_execution_results"] = gold_exec_results

            result_entry["model_output_raw"] = pred_text

            # Add database info if available
            if "db_id" in item:
                result_entry["db_id"] = item["db_id"]

            results["detailed_results"].append(result_entry)

            # Write to detailed log file if specified
            if detailed_log_file:
                detailed_log_file.write(f"{'='*100}\n")
                detailed_log_file.write(f"Sample {idx + 1}/{len(eval_samples)}\n")
                detailed_log_file.write(f"Question ID: {item.get('question_id', f'sample_{idx}')}\n")
                detailed_log_file.write(f"Database: {item.get('db_id', 'N/A')}\n")
                detailed_log_file.write(f"Question: {item.get('question', 'N/A')}\n")
                detailed_log_file.write(f"{'-'*50}\n")
                detailed_log_file.write(f"PROMPT:\n{prompt}\n")
                detailed_log_file.write(f"{'-'*50}\n")

                if task_type == 'ra':
                    detailed_log_file.write(f"GROUND TRUTH (RA):\n{gold_ra_str}\n")
                    detailed_log_file.write(f"{'-'*50}\n")
                    detailed_log_file.write(f"MODEL OUTPUT (RAW):\n{pred_text}\n")
                    detailed_log_file.write(f"{'-'*50}\n")
                    detailed_log_file.write(f"Score: {cont_res['score']}\n")
                    detailed_log_file.write(f"Component Recall Score: {cont_res['component_recall_score']}\n")
                    detailed_log_file.write(f"{'-'*50}\n")
                    if pred_ra is not None:
                        detailed_log_file.write(f"MODEL OUTPUT (PARSED):\n{json.dumps(pred_ra, indent=2)}\n")
                    else:
                        detailed_log_file.write(f"MODEL OUTPUT (PARSED): Failed to parse JSON\n")
                elif task_type == 'ra_sql':
                    # Write both RA and SQL ground truth and predictions
                    if gold_ra is not None:
                        detailed_log_file.write(f"GROUND TRUTH (RA):\n{json.dumps(gold_ra, indent=2)}\n")
                        detailed_log_file.write(f"{'-'*50}\n")
                    detailed_log_file.write(f"GROUND TRUTH (SQL):\n{gold_sql}\n")
                    detailed_log_file.write(f"{'-'*50}\n")
                    detailed_log_file.write(f"MODEL OUTPUT (RAW):\n{pred_text}\n")
                    detailed_log_file.write(f"{'-'*50}\n")
                    if pred_ra is not None:
                        detailed_log_file.write(f"PREDICTED RA:\n{json.dumps(pred_ra, indent=2)}\n")
                        detailed_log_file.write(f"RA Match: {ra_match}\n")
                    else:
                        detailed_log_file.write(f"PREDICTED RA: Failed to parse\n")
                    detailed_log_file.write(f"{'-'*50}\n")
                    if pred_sql is not None:
                        detailed_log_file.write(f"PREDICTED SQL:\n{pred_sql}\n")
                        detailed_log_file.write(f"SQL Match: {sql_match}\n")
                    else:
                        detailed_log_file.write(f"PREDICTED SQL: Failed to parse\n")
                    detailed_log_file.write(f"{'-'*50}\n")
                    detailed_log_file.write(f"RA Score: {cont_res.get('ra_score', 0)}\n")
                    detailed_log_file.write(f"SQL Score: {cont_res.get('sql_score', 0)}\n")
                    detailed_log_file.write(f"Combined Score: {cont_res['score']}\n")
                    if exec_accuracy is not None:
                        detailed_log_file.write(f"Execution Accuracy: {exec_accuracy}\n")
                else:  # SQL
                    detailed_log_file.write(f"GROUND TRUTH (SQL):\n{gold_sql}\n")
                    detailed_log_file.write(f"{'-'*50}\n")
                    detailed_log_file.write(f"MODEL OUTPUT (RAW):\n{pred_text}\n")
                    detailed_log_file.write(f"{'-'*50}\n")
                    detailed_log_file.write(f"PREDICTED SQL:\n{pred_sql if pred_sql else 'Failed to parse'}\n")
                    detailed_log_file.write(f"{'-'*50}\n")
                    detailed_log_file.write(f"Score: {cont_res['score']}\n")
                    if exec_accuracy is not None:
                        detailed_log_file.write(f"Execution Accuracy: {exec_accuracy}\n")

                if error_message:
                    detailed_log_file.write(f"Error: {error_message}\n")
                detailed_log_file.write(f"{'-'*50}\n")
                detailed_log_file.write(f"RESULT: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")
                detailed_log_file.write(f"Generation Time: {gen_time:.3f}s\n")
                detailed_log_file.write(f"\n")
                detailed_log_file.flush()  # Flush after each sample for real-time monitoring

            # Print verbose output
            if verbose:
                # Determine if we should print this sample
                should_print = False
                if verbose == True:  # Standard verbose: first 10 and all errors
                    should_print = (idx < 10) or not is_correct
                elif verbose == 2:  # Very verbose: all samples
                    should_print = True

                if should_print:
                    print(f"\n{'='*80}")
                    print(f"Sample {idx + 1}/{len(eval_samples)} | DB: {item.get('db_id', 'N/A')}")
                    print(f"Question ID: {item.get('question_id', f'sample_{idx}')}")
                    print(f"Question: {item.get('question', 'N/A')}")
                    print(f"-" * 40)

                    # Show prompt if very verbose
                    if verbose == 2:
                        print(f"Prompt (first 500 chars):")
                        print(f"  {prompt[:500]}...")
                        print(f"-" * 40)

                    # Always show ground truth
                    print(f"Ground Truth:")
                    if task_type == 'ra':
                        print(f"  {gold_ra_str}")
                    elif task_type == 'ra_sql':
                        if gold_ra is not None:
                            print(f"  RA: {json.dumps(gold_ra, indent=2)}")
                        print(f"  SQL: {gold_sql}")
                    else:  # SQL
                        print(f"  {gold_sql}")
                    print(f"-" * 40)

                    # Show model output
                    print(f"Model Output (Raw):")
                    if len(pred_text) > 500:
                        print(f"  {pred_text[:500]}...")
                        print(f"  [... truncated, total length: {len(pred_text)} chars]")
                    else:
                        print(f"  {pred_text}")
                    print(f"-" * 40)

                    # Show parsed output
                    if task_type == 'ra':
                        if pred_ra is not None:
                            print(f"Model Output (Parsed):")
                            print(f"  {json.dumps(pred_ra, indent=2)}")
                        else:
                            print(f"Model Output (Parsed): Failed to parse RA JSON")
                    elif task_type == 'ra_sql':
                        if pred_ra is not None:
                            print(f"Predicted RA:")
                            print(f"  {json.dumps(pred_ra, indent=2)}")
                            print(f"  RA Match: {ra_match}")
                        else:
                            print(f"Predicted RA: Failed to parse")
                        if pred_sql is not None:
                            print(f"Predicted SQL:")
                            print(f"  {pred_sql}")
                            print(f"  SQL Match: {sql_match}")
                        else:
                            print(f"Predicted SQL: Failed to parse")
                    else:  # SQL
                        if pred_sql is not None:
                            print(f"Predicted SQL:")
                            print(f"  {pred_sql}")
                        else:
                            print(f"Predicted SQL: Failed to parse SQL")
                    print(f"-" * 40)

                    # Show result
                    print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                    if error_message:
                        print(f"Error: {error_message}")
                    print(f"Generation Time: {gen_time:.3f}s")

        # Save log of RA pairs if specified
        if log_ra_pair_path and debug:
            with open(log_ra_pair_path, 'w', encoding='utf-8') as f:
                json.dump(log_json, f, indent=2)
            print(f"Saved RA pairs log to: {log_ra_pair_path}")

        # Calculate final metrics
        results["overall_metrics"]["accuracy"] = (
            results["overall_metrics"]["correct"] / results["overall_metrics"]["total"]
            if results["overall_metrics"]["total"] > 0 else 0.0
        )
        results["overall_metrics"]["avg_generation_time"] = (
            total_gen_time / results["overall_metrics"]["total"]
            if results["overall_metrics"]["total"] > 0 else 0.0
        )
        results['overall_metrics']['avg_score'] = (
            results['overall_metrics']['score'] / results['overall_metrics']['total']
            if results['overall_metrics']['total'] > 0 else 0.0
        )
        results['overall_metrics']['avg_component_recall_score'] = (
            results['overall_metrics']['component_recall_score'] / results['overall_metrics']['total']
            if results['overall_metrics']['total'] > 0 else 0.0
        )
        results["overall_metrics"]["total_generation_time"] = total_gen_time

        # Calculate SQL-specific metrics if applicable
        if task_type in ['sql', 'ra_sql', 'cot']:
            results["overall_metrics"]["exec_accuracy"] = (
                results["overall_metrics"]["exec_correct"] / results["overall_metrics"]["total"]
                if results["overall_metrics"]["total"] > 0 else 0.0
            )
            # Calculate BIRD-style metrics
            results["overall_metrics"]["exec_bird_accuracy"] = (
                results["overall_metrics"]["exec_bird_correct"] / results["overall_metrics"]["total"]
                if results["overall_metrics"]["total"] > 0 else 0.0
            )
            # Calculate Official Spider metrics
            results["overall_metrics"]["exec_spider_official_accuracy"] = (
                results["overall_metrics"]["exec_spider_official_correct"] / results["overall_metrics"]["total"]
                if results["overall_metrics"]["total"] > 0 else 0.0
            )

        # Calculate RA+SQL specific metrics
        if task_type == 'ra_sql':
            total = results["overall_metrics"]["total"]
            if total > 0:
                results["overall_metrics"]["ra_accuracy"] = results["overall_metrics"]["ra_correct"] / total
                results["overall_metrics"]["sql_accuracy"] = results["overall_metrics"]["sql_correct"] / total
                results["overall_metrics"]["both_accuracy"] = results["overall_metrics"]["both_correct"] / total
                results["overall_metrics"]["avg_ra_score"] = results["overall_metrics"]["ra_score"] / total
                results["overall_metrics"]["avg_sql_score"] = results["overall_metrics"]["sql_score"] / total
                results["overall_metrics"]["avg_ra_component_recall_score"] = results["overall_metrics"]["ra_component_recall_score"] / total
                results["overall_metrics"]["avg_sql_component_recall_score"] = results["overall_metrics"]["sql_component_recall_score"] / total

            # Clean up raw scores
            del results['overall_metrics']['ra_score']
            del results['overall_metrics']['sql_score']
            del results['overall_metrics']['ra_component_recall_score']
            del results['overall_metrics']['sql_component_recall_score']



        del results['overall_metrics']['score']
        del results['overall_metrics']['component_recall_score']

        # Close detailed log file with summary
        if detailed_log_file:
            detailed_log_file.write(f"\n{'='*100}\n")
            detailed_log_file.write(f"EVALUATION SUMMARY\n")
            detailed_log_file.write(f"{'='*100}\n")
            detailed_log_file.write(f"Total Samples: {results['overall_metrics']['total']}\n")
            detailed_log_file.write(f"Correct: {results['overall_metrics']['correct']}\n")
            detailed_log_file.write(f"Incorrect: {results['overall_metrics']['total'] - results['overall_metrics']['correct']}\n")
            detailed_log_file.write(f"Accuracy: {results['overall_metrics']['accuracy']*100:.2f}%\n")
            detailed_log_file.write(f"Parse Errors: {results['overall_metrics']['parse_errors']}\n")
            detailed_log_file.write(f"Logic Errors: {results['overall_metrics']['total'] - results['overall_metrics']['correct'] - results['overall_metrics']['parse_errors']}\n")
            detailed_log_file.write(f"Average Generation Time: {results['overall_metrics']['avg_generation_time']:.3f}s\n")
            detailed_log_file.write(f"Total Generation Time: {results['overall_metrics']['total_generation_time']:.1f}s\n")
            if results['overall_metrics']['total_generation_time'] > 0:
                throughput = results['overall_metrics']['total'] / results['overall_metrics']['total_generation_time']
                detailed_log_file.write(f"Throughput: {throughput:.2f} samples/second\n")
            detailed_log_file.close()
            print(f"\nDetailed log saved to: {detailed_log_path}")

        return results

    def save_results(self, results: Dict[str, Any], output_path: str, save_errors_separately: bool = False):
        """Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save main results
            save_errors_separately: If True, also save errors to a separate file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

        # Save errors separately if requested
        if save_errors_separately:
            errors = [r for r in results["detailed_results"] if not r["is_correct"]]
            if errors:
                error_path = output_path.replace('.json', '_errors.json')
                error_summary = {
                    "metadata": results["metadata"],
                    "error_statistics": {
                        "total_errors": len(errors),
                        "parse_errors": sum(1 for e in errors if e["parse_error"]),
                        "logic_errors": sum(1 for e in errors if not e["parse_error"]),
                        "error_rate": len(errors) / results["overall_metrics"]["total"] * 100
                    },
                    "errors": errors
                }
                with open(error_path, 'w') as f:
                    json.dump(error_summary, f, indent=2)
                print(f"Error analysis saved to {error_path}")

    def print_summary(self, results: Dict[str, Any], show_sample_errors: int = 5):
        """Print evaluation summary.

        Args:
            results: Evaluation results
            show_sample_errors: Number of sample errors to display (0 to disable)
        """
        metrics = results["overall_metrics"]
        task_type = results["metadata"].get("task_type", "ra")

        print("\n" + "="*80)
        print(f"EVALUATION SUMMARY (vLLM, Chat-Aligned) - {task_type.upper()} Task")
        print("="*80)
        print(f"Total Samples: {metrics['total']}")

        if task_type == 'ra_sql':
            print(f"\nRA Metrics:")
            print(f"  RA Correct: {metrics['ra_correct']}")
            print(f"  RA Accuracy: {metrics['ra_accuracy']*100:.2f}%")
            print(f"  Avg RA Score: {metrics['avg_ra_score']:.4f}")
            print(f"  Avg RA Component Recall Score: {metrics['avg_ra_component_recall_score']:.4f}")

            print(f"\nSQL Metrics:")
            print(f"  SQL Correct: {metrics['sql_correct']}")
            print(f"  SQL Accuracy: {metrics['sql_accuracy']*100:.2f}%")
            print(f"  Avg SQL Score: {metrics['avg_sql_score']:.4f}")
            print(f"  Avg SQL Component Recall Score: {metrics['avg_sql_component_recall_score']:.4f}")

            print(f"\nCombined Metrics:")
            print(f"  Both Correct: {metrics['both_correct']}")
            print(f"  Both Accuracy: {metrics['both_accuracy']*100:.2f}%")

            if 'exec_accuracy' in metrics:
                print(f"\n  Execution Metrics:")
                print(f"    Spider res_map:         {metrics['exec_accuracy']*100:.2f}% ({metrics['exec_correct']}/{metrics['total']})")
                if 'exec_bird_accuracy' in metrics:
                    print(f"    BIRD-style (official):  {metrics['exec_bird_accuracy']*100:.2f}% ({metrics['exec_bird_correct']}/{metrics['total']})")
                if 'exec_spider_official_accuracy' in metrics:
                    print(f"    Spider Official:        {metrics['exec_spider_official_accuracy']*100:.2f}% ({metrics['exec_spider_official_correct']}/{metrics['total']})")
                if 'exec_bird_accuracy' in metrics and 'exec_accuracy' in metrics:
                    diff = (metrics['exec_bird_accuracy'] - metrics['exec_accuracy']) * 100
                    if diff > 0:
                        print(f"    BIRD vs res_map:        +{diff:.2f}% (BIRD more lenient)")
                    elif diff < 0:
                        print(f"    BIRD vs res_map:        {diff:.2f}% (res_map more lenient)")
                if 'exec_spider_official_accuracy' in metrics and 'exec_accuracy' in metrics:
                    diff2 = (metrics['exec_spider_official_accuracy'] - metrics['exec_accuracy']) * 100
                    if diff2 > 0:
                        print(f"    Official vs res_map:    +{diff2:.2f}% (Official more lenient)")
                    elif diff2 < 0:
                        print(f"    Official vs res_map:    {diff2:.2f}% (res_map more lenient)")
        else:
            print(f"Correct: {metrics['correct']}")
            print(f"Incorrect: {metrics['total'] - metrics['correct']}")
            print(f"Accuracy: {metrics['accuracy']*100:.2f}%")

            if task_type in ['sql', 'cot'] and 'exec_accuracy' in metrics:
                print(f"\nExecution Metrics:")
                print(f"  Spider res_map:         {metrics['exec_accuracy']*100:.2f}% ({metrics['exec_correct']}/{metrics['total']})")
                if 'exec_bird_accuracy' in metrics:
                    print(f"  BIRD-style (official):  {metrics['exec_bird_accuracy']*100:.2f}% ({metrics['exec_bird_correct']}/{metrics['total']})")
                if 'exec_spider_official_accuracy' in metrics:
                    print(f"  Spider Official:        {metrics['exec_spider_official_accuracy']*100:.2f}% ({metrics['exec_spider_official_correct']}/{metrics['total']})")
                if 'exec_bird_accuracy' in metrics and 'exec_accuracy' in metrics:
                    diff = (metrics['exec_bird_accuracy'] - metrics['exec_accuracy']) * 100
                    if diff > 0:
                        print(f"  BIRD vs res_map:        +{diff:.2f}% (BIRD more lenient)")
                    elif diff < 0:
                        print(f"  BIRD vs res_map:        {diff:.2f}% (res_map more lenient)")
                    else:
                        print(f"  BIRD vs res_map:        0.00% (identical)")
                if 'exec_spider_official_accuracy' in metrics and 'exec_accuracy' in metrics:
                    diff2 = (metrics['exec_spider_official_accuracy'] - metrics['exec_accuracy']) * 100
                    if diff2 > 0:
                        print(f"  Official vs res_map:    +{diff2:.2f}% (Official more lenient)")
                    elif diff2 < 0:
                        print(f"  Official vs res_map:    {diff2:.2f}% (res_map more lenient)")
                    else:
                        print(f"  Official vs res_map:    0.00% (identical)")
                    if metrics.get('exec_methods_agree'):
                        agree_rate = metrics['exec_methods_agree'] / metrics['total'] * 100
                        print(f"  Methods agree:          {agree_rate:.2f}% ({metrics['exec_methods_agree']}/{metrics['total']})")

            print(f"Avg Score: {metrics['avg_score']:.4f}")
            print(f"Avg Component Recall Score: {metrics['avg_component_recall_score']:.4f}")

        print(f"Parse Errors: {metrics['parse_errors']} ({metrics['parse_errors']/metrics['total']*100:.1f}%)")
        print(f"Average Generation Time: {metrics['avg_generation_time']:.3f}s")
        print(f"Total Generation Time: {metrics['total_generation_time']:.1f}s")

        # Calculate throughput
        if metrics['total_generation_time'] > 0:
            throughput = metrics['total'] / metrics['total_generation_time']
            print(f"Throughput: {throughput:.2f} samples/second")

        # Analyze errors
        if results["detailed_results"]:
            errors = [r for r in results["detailed_results"] if not r["is_correct"]]
            if errors:
                print(f"\n" + "-"*80)
                print("ERROR ANALYSIS")
                print("-"*80)
                print(f"Total Errors: {len(errors)}")
                print(f"Parse Errors: {sum(1 for e in errors if e['parse_error'])}")
                print(f"Logic Errors: {sum(1 for e in errors if not e['parse_error'])}")

                # Show sample errors if requested
                if show_sample_errors > 0 and errors:
                    print(f"\nSample Errors (showing first {min(show_sample_errors, len(errors))}):")
                    for i, err in enumerate(errors[:show_sample_errors]):
                        print(f"\n  Error {i+1}:")
                        print(f"    Question ID: {err['question_id']}")
                        print(f"    Question: {err['question'][:100]}...")
                        print(f"    Error Type: {'Parse Error' if err['parse_error'] else 'Logic Error'}")
                        if err.get('error_message'):
                            print(f"    Message: {err['error_message']}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Chat-Aligned Relational Algebra or SQL SFT Model with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RA evaluation
  %(prog)s --model_checkpoint_path ./ra_chat_model --input_file test_ra.json \\
           --database_path ./db --dataset_name spider \\
           --table_value_cache_path ./cache --table_info_cache_path ./info \\
           --task_type ra

  # SQL evaluation with Spider metrics (with foreign keys)
  %(prog)s --model_checkpoint_path ./sql_chat_model --input_file test_sql.json \\
           --database_path ./db --dataset_name spider \\
           --table_value_cache_path ./cache --table_info_cache_path ./info \\
           --table_json_path ./tables.json --task_type sql

  # SQL evaluation for BIRD (no tables.json needed)
  %(prog)s --model_checkpoint_path ./sql_chat_model --input_file bird_sql.json \\
           --database_path ./bird_db --dataset_name bird \\
           --table_value_cache_path ./cache --table_info_cache_path ./info \\
           --task_type sql

  # Show first 10 samples and all errors
  %(prog)s --model_checkpoint_path ./model --input_file test.json \\
           --verbose 1 ...

  # Show all samples with detailed output
  %(prog)s --model_checkpoint_path ./model --input_file test.json \\
           --verbose 2 --detailed_log eval_log.txt ...

  # Use custom temp directory instead of /tmp
  %(prog)s --model_checkpoint_path ./model --input_file test.json \\
           --temp_dir_base ./my_temp_dir ...

  # For checkpoint without config.json
  %(prog)s --model_checkpoint_path ./checkpoint-1251 \\
           --tokenizer_name Qwen/Qwen2.5-0.5B-Instruct \\
           --temp_dir_base ./vllm_temp ...
        """
    )

    # Model arguments
    parser.add_argument("--model_checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer/base model name if checkpoint lacks config (e.g., Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--system_message", type=str, default=DEFAULT_SYSTEM_MESSAGE,
                        help="System message for chat template")

    # vLLM arguments
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization fraction")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Maximum model context length")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for vLLM inference")

    # Dataset arguments
    parser.add_argument("--database_path", type=str, required=True,
                        help="Path to database")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name (e.g., spider)")
    parser.add_argument("--table_value_cache_path", type=str, required=True,
                        help="Path to table value cache")
    parser.add_argument("--table_info_cache_path", type=str, required=True,
                        help="Path to table info cache")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to evaluation dataset JSON file")

    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p for nucleus sampling")

    # Evaluation arguments
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--output_log", type=str, default="evaluation_results_vllm_chat.json",
                        help="Path to save detailed evaluation results")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2],
                        help="Verbosity level: 0=summary only, 1=show first 10 and errors, 2=show all")
    parser.add_argument("--detailed_log", type=str, default=None,
                        help="Path to save detailed text log with all outputs")
    parser.add_argument("--save_errors", action="store_true",
                        help="Save errors to a separate JSON file for analysis")
    parser.add_argument("--exec_timeout", type=int, default=None,
                        help="SQL execution timeout in seconds (default: None = no timeout). Set to 60-120 for complex queries that may hang.")
    parser.add_argument("--save_exec_details", action="store_true",
                        help="Save per-sample execution details (both Spider and BIRD metrics) for gap analysis")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--temp_dir_base", type=str, default=None,
                        help="Base directory for temporary files (default: current directory, not /tmp)")
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--mode', type=str, default='dev', choices=['train', 'dev'],
                        help='Mode for schema generation: train (partial columns) or dev (all columns)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose error outputs')
    parser.add_argument('--log_ra_pair_path', type=str, default=None,
                        help='Path to log relational algebra and parse analysis pairs (if needed)')

    # SQL evaluation specific arguments
    parser.add_argument('--task_type', type=str, default='ra', choices=['ra', 'sql', 'ra_sql', 'cot'],
                        help='Task type: "ra" for relational algebra, "sql" for SQL evaluation, "ra_sql" for combined RA+SQL evaluation, "cot" for chain-of-thought SQL evaluation')
    parser.add_argument('--skip_ra_eval', action='store_true',
                        help='Skip RA evaluation in ra_sql mode (only evaluate SQL). Has no effect for "ra" or "sql" task types. Default: False (backward compatible)')
    parser.add_argument('--table_json_path', type=str, default=None,
                        help='Path to tables.json for SQL evaluation (optional, improves Spider evaluation accuracy)')

    args = parser.parse_args()

    # Warn if table_json_path not provided for SQL, RA+SQL, or CoT evaluation
    if args.task_type in ['sql', 'ra_sql', 'cot'] and not args.table_json_path:
        print(f"Warning: --table_json_path not provided for {args.task_type.upper()} evaluation.")
        print("         This is fine for BIRD dataset, but may reduce accuracy for Spider.")
        print("         Foreign key relationships will not be considered in evaluation.")

    # Configure temp directory for vLLM before any imports/initialization
    if args.temp_dir_base:
        os.makedirs(args.temp_dir_base, exist_ok=True)
    configure_vllm_temp_dir(args.temp_dir_base or os.getcwd())

    # Adjust output file names based on task type and COT
    if args.task_type == 'sql':
        args.output_log = args.output_log.replace('evaluation_results', 'sql_evaluation_results')
        if args.detailed_log:
            args.detailed_log = args.detailed_log.replace('.txt', '_sql.txt')
    elif args.task_type == 'ra_sql':
        args.output_log = args.output_log.replace('evaluation_results', 'ra_sql_evaluation_results')
        if args.detailed_log:
            args.detailed_log = args.detailed_log.replace('.txt', '_ra_sql.txt')
    elif args.task_type == 'cot':
        args.output_log = args.output_log.replace('evaluation_results', 'cot_evaluation_results')
        if args.detailed_log:
            args.detailed_log = args.detailed_log.replace('.txt', '_cot.txt')

    if args.cot:
        args.output_log = args.output_log.replace('.json', '_cot.json')
        args.detailed_log = None if args.detailed_log is None else args.detailed_log.replace('.txt', '_cot.txt')

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load evaluation dataset
    print(f"Loading evaluation dataset from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        eval_dataset = json.load(f)
    print(f"Loaded {len(eval_dataset)} samples")

    # Initialize evaluator
    evaluator = ChatAlignedRelationalAlgebraVLLMEvaluator(
        model_checkpoint_path=args.model_checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        temp_dir_base=args.temp_dir_base,
        system_message=args.system_message,
    )

    # Run evaluation
    print(f"\nStarting {args.task_type.upper()} evaluation with vLLM (chat-aligned)...")
    print(f"Verbosity level: {args.verbose} (0=summary, 1=first 10 + errors, 2=all)")
    if args.detailed_log:
        print(f"Detailed log will be saved to: {args.detailed_log}")
    if args.task_type in ['sql', 'cot']:
        print(f"Using tables.json from: {args.table_json_path}")
    if args.task_type == 'cot':
        print(f"Note: CoT models output reasoning + SQL. SQL will be extracted for evaluation.")

    results = evaluator.evaluate_dataset(
        dataset=eval_dataset,
        database_path=args.database_path,
        dataset_name=args.dataset_name,
        table_value_cache_path=args.table_value_cache_path,
        table_info_cache_path=args.table_info_cache_path,
        model_checkpoint_path=args.model_checkpoint_path,
        tokenizer_name=args.tokenizer_name,
        max_samples=args.max_samples,
        verbose=args.verbose,
        detailed_log_path=args.detailed_log,
        cot=args.cot,
        mode=args.mode,
        debug=args.debug,
        log_ra_pair_path=args.log_ra_pair_path,
        task_type=args.task_type,
        table_json_path=args.table_json_path,
        save_exec_details=args.save_exec_details,
        exec_timeout=args.exec_timeout,
        skip_ra_eval=args.skip_ra_eval,
    )

    # Save results
    evaluator.save_results(results, args.output_log, save_errors_separately=args.save_errors)

    # Print summary
    evaluator.print_summary(results, show_sample_errors=5 if args.verbose > 0 else 0)

    # Create a simplified CSV report if needed
    csv_path = args.output_log.replace('.json', '_summary.csv')
    print(f"\nCreating summary CSV at {csv_path}...")

    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Index', 'Question_ID', 'DB_ID', 'Question',
            'Is_Correct', 'Parse_Error', 'Generation_Time',
            'Ground_Truth_Summary', 'Model_Output_Summary'
        ])
        for r in results['detailed_results']:
            # Create summaries of ground truth and model output
            if 'ground_truth' in r:
                gt_summary = str(r['ground_truth']).replace('\n', ' ')
            elif 'ground_truth_ra' in r and 'ground_truth_sql' in r:
                # For ra_sql task type
                gt_summary = f"RA+SQL task"
            elif 'ground_truth_sql' in r:
                gt_summary = str(r['ground_truth_sql']).replace('\n', ' ')[:200]  # Limit SQL length
            else:
                gt_summary = 'N/A'

            if 'model_output_parsed' in r:
                model_summary = str(r['model_output_parsed']).replace('\n', ' ')
            elif 'predicted_ra' in r and 'predicted_sql' in r:
                # For ra_sql task type, show both RA and SQL status
                ra_status = 'RA OK' if r.get('ra_match', False) else 'RA Failed'
                sql_status = 'SQL OK' if r.get('sql_match', False) else 'SQL Failed'
                model_summary = f'{ra_status}, {sql_status}'
            elif 'predicted_sql' in r:
                model_summary = str(r['predicted_sql']).replace('\n', ' ')[:200] if r['predicted_sql'] else 'Parse Failed'
            else:
                model_summary = 'N/A'

            writer.writerow([
                r['index'],
                r['question_id'],
                r.get('db_id', 'N/A'),
                r.get('question', ''),
                r['is_correct'],
                r['parse_error'],
                f"{r['generation_time_seconds']:.3f}",
                gt_summary,
                model_summary
            ])

    print(f"Summary CSV saved to {csv_path}")
    print("\nEvaluation complete!")

    # Clean up temp directory if created
    if hasattr(evaluator, 'temp_dir') and evaluator.temp_dir:
        print(f"Cleaning up temporary directory: {evaluator.temp_dir}")
        try:
            shutil.rmtree(evaluator.temp_dir)
            print("Temporary directory cleaned up successfully")
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")


if __name__ == "__main__":
    main()
