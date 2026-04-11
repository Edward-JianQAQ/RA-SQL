#!/usr/bin/env python
"""
Evaluate a trained RA or SQL SFT model on a provided test split.

Example for RA evaluation (BIRD):
python ra_eval.py \
  --model_checkpoint_path=ra_sql_ckpts/ra_Qwen2.5-7B_sft_model \
  --database_path=ra_data/bird/dev_20240627/dev_databases \
  --dataset_name=bird \
  --table_value_cache_path=ra_data/bird/dev_20240627/bird_dev_id2sampled_values.json \
  --table_info_cache_path=ra_data/bird/dev_20240627/bird_dev_id2db_info.json \
  --input_file=ra_data/bird/dev_20240627/dev_bird_ra_correct.json \
  --tokenizer_name=Qwen/Qwen2.5-3B \
  --detailed_log evaluation_detailed.txt \
  --verbose 0 \
  --save_errors \
  --cot \
  --mode dev \
  --task_type ra
 
Example for RA evaluation (Spider): 
python ra_eval.py \
    --model_checkpoint_path=ra_sql_ckpts/ra_Qwen2.5-3B_ra_sql_ckpts/spider_ra_sft_model_cot_pre_order_3epoch_cot_pre_order \
    --database_path=ra_data/spider/database \
    --dataset_name=spider \
    --table_value_cache_path=ra_data/spider/spider_dev_id2sampled_values.json \
    --table_info_cache_path=ra_data/spider/spider_dev_id2db_info.json \
    --input_file=ra_data/spider/dev_spider_ra_correct.json \
    --table_json_path=ra_data/spider/tables.json \
    --tokenizer_name=Qwen/Qwen2.5-3B \
    --detailed_log sql_evaluation_detailed.txt \
    --verbose 0 \
    --save_errors \
    --cot \
    --mode dev \
    --task_type ra

Example for SQL evaluation (Spider with foreign keys):
python ra_eval.py \
  --model_checkpoint_path=sql_sft_model \
  --database_path=ra_data/spider/database \
  --dataset_name=spider \
  --table_value_cache_path=ra_data/spider/spider_dev_id2sampled_values.json \
  --table_info_cache_path=ra_data/spider/spider_dev_id2db_info.json \
  --input_file=ra_data/bird/dev_20240627/dev_bird_ra_correct.json \
  --table_json_path=ra_data/spider/tables.json \
  --tokenizer_name=Qwen/Qwen2.5-3B \
  --detailed_log sql_evaluation_detailed.txt \
  --verbose 0 \
  --save_errors \
  --cot \
  --mode dev \
  --task_type sql

Example for SQL evaluation (BIRD without tables.json):
python ra_eval.py \
  --model_checkpoint_path=sql_sft_model \
  --database_path=ra_data/bird/dev_20240627/dev_databases \
  --dataset_name=bird \
  --table_value_cache_path=ra_data/bird/dev_20240627/bird_dev_id2sampled_values.json \
  --table_info_cache_path=ra_data/bird/dev_20240627/bird_dev_id2db_info.json \
  --input_file=ra_data/bird/dev_20240627/dev_bird_ra_correct.json \
  --tokenizer_name=Qwen/Qwen2.5-3B \
  --detailed_log sql_bird_evaluation.txt \
  --verbose 0 \
  --task_type sql

Example for RA+SQL evaluation:
python ra_eval.py \
  --model_checkpoint_path=ra_sql_sft_model \
  --database_path=ra_data/bird/dev_20240627/dev_databases \
  --dataset_name=bird \
  --table_value_cache_path=ra_data/bird/dev_20240627/bird_dev_id2sampled_values.json \
  --table_info_cache_path=ra_data/bird/dev_20240627/bird_dev_id2db_info.json \
  --input_file=ra_data/bird/dev_20240627/dev_bird_ra_correct.json \
  --tokenizer_name=Qwen/Qwen2.5-3B \
  --detailed_log ra_sql_evaluation.txt \
  --verbose 0 \
  --task_type ra_sql
"""

"""
Evaluation script for Relational Algebra, SQL, or combined RA+SQL SFT model using vLLM for fast inference.
Performs detailed evaluation and logs results.

Supports three task types:
- 'ra': Evaluates Relational Algebra JSON outputs
- 'sql': Evaluates SQL query outputs using Spider evaluation metrics
- 'ra_sql': Evaluates both RA (from thinking) and SQL (from answer) outputs

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
python ra_eval.py \
    --model_checkpoint_path ./ra_sft_model \
    --database_path /path/to/database \
    --dataset_name spider \
    --table_value_cache_path /path/to/cache \
    --table_info_cache_path /path/to/info_cache \
    --input_file /path/to/test_data.json \
    --output_log evaluation_results.json \
    --task_type ra

Example usage for SQL:
python ra_eval.py \
    --model_checkpoint_path ./sql_sft_model \
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
# import pdb  # Commented out, uncomment for debugging

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


class RelationalAlgebraVLLMEvaluator:
    """Evaluator for Relational Algebra and SQL models using vLLM for fast inference."""
    
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
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.model_checkpoint_path = model_checkpoint_path
        
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
        
        # Check if checkpoint has necessary files
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
        
        if not has_config and tokenizer_name:
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
        if not _has_standard_weights(model_checkpoint_path):
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
    
    def generate_batch_predictions(self, prompts: List[str]) -> List[tuple[str, float]]:
        """Generate predictions for a batch of prompts using vLLM."""
        start_time = time.time()
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            stop=[self.tokenizer.eos_token] if self.tokenizer.eos_token else None,
        )
        
        # Add newline to prompts
        formatted_prompts = [p + "\n" for p in prompts]
        
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
        table_json_path: Optional[str] = None
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

        Returns:
            Dictionary containing evaluation results with metrics and detailed outputs
        """

        # Load foreign key maps if evaluating SQL or RA+SQL (optional, mainly for Spider)
        kmaps = None
        if task_type in ['sql', 'ra_sql'] and table_json_path and os.path.exists(table_json_path):
            print(f"Loading foreign key maps from {table_json_path}...")
            try:
                kmaps = build_foreign_key_map_from_json(table_json_path)
            except Exception as e:
                print(f"Warning: Could not load foreign key maps: {e}")
                print("Continuing without foreign key maps (may affect accuracy slightly)")
                kmaps = None
        
        results = {
            "metadata": {
                "model_checkpoint": model_checkpoint_path,
                "tokenizer": tokenizer_name or model_checkpoint_path,
                "evaluation_timestamp": datetime.now().isoformat(),
                "num_samples": min(len(dataset), max_samples) if max_samples else len(dataset),
                "task_type": task_type,
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

        # Add SQL-specific metrics if evaluating SQL or RA+SQL
        if task_type in ['sql', 'ra_sql']:
            results["overall_metrics"]["exec_accuracy"] = 0.0
            results["overall_metrics"]["exec_correct"] = 0

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
            detailed_log_file.write(f"Evaluation Log - {datetime.now().isoformat()}\n")
            detailed_log_file.write(f"Model: {model_checkpoint_path}\n")
            detailed_log_file.write(f"Tokenizer: {tokenizer_name or model_checkpoint_path}\n")
            detailed_log_file.write(f"Dataset: {dataset_name}\n")
            detailed_log_file.write(f"Total Samples to Evaluate: {len(eval_samples)}\n")
            detailed_log_file.write(f"{'='*100}\n\n")

        if log_ra_pair_path and debug:
            log_json = []
        
        for idx, (item, prompt, (pred_text, gen_time)) in enumerate(
            tqdm(zip(batch_items, batch_prompts, all_predictions), desc="Processing", total=len(batch_items))
        ):
            results["overall_metrics"]["total"] += 1
            total_gen_time += gen_time

            # Initialize common variables
            is_correct = False
            parse_error = False
            error_message = None
            cont_res = {
                'score': 0,
                'component_recall_score': 0
            }
            exec_accuracy = None  # For SQL evaluation

            if task_type == 'ra':
                # Get ground truth RA
                gold_ra = item["relational_algebra"]
                gold_ra_str = json.dumps(gold_ra, indent=2)

                # Parse prediction
                pred_ra = self.parse_ra_output(pred_text)

                # if log_ra_pair_path and debug:
                #     # save as a json dict
                #     log_entry = {
                #         "question_id": item.get("question_id", f"sample_{idx}"),
                #         "question": item.get("question", ""),
                #         "SQL": item.get("SQL", ""),
                #         "ground_truth": gold_ra,
                #         "model_output_raw": pred_text,
                #         "model_output_parsed": pred_ra
                #     }
                #     log_json.append(log_entry)

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
                # RA+SQL evaluation - evaluate both components

                # Get ground truth
                gold_ra = item["relational_algebra"]
                gold_sql = get_sql_field_from_item(item, dataset_name)
                db_id = item.get("db_id", "")

                # Initialize tracking variables
                ra_match = False
                sql_match = False
                ra_cont_res = {'score': 0, 'component_recall_score': 0}
                sql_cont_res = {'score': 0, 'component_recall_score': 0}

                # Extract and evaluate RA from thinking section
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
                else:
                    # Evaluate SQL using Spider evaluation
                    try:
                        if kmaps is not None:
                            eval_result = evaluate_single_pair(
                                pred_sql=pred_sql,
                                gold_sql=gold_sql,
                                db_dir=database_path,
                                db_name=db_id,
                                etype='all',
                                kmaps=kmaps
                            )
                        else:
                            # Create empty kmaps dict for evaluation
                            empty_kmaps = {db_id: {}} if db_id else {}
                            eval_result = evaluate_single_pair(
                                pred_sql=pred_sql,
                                gold_sql=gold_sql,
                                db_dir=database_path,
                                db_name=db_id,
                                etype='all',
                                kmaps=empty_kmaps
                            )

                        # Extract SQL scores
                        sql_match = eval_result['exact'] == 1
                        exec_accuracy = eval_result['exec']

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
                    except Exception as e:
                        # Fallback to simple string comparison
                        if debug:
                            print(f"Warning: Full SQL evaluation failed, using string comparison: {e}")
                        gold_normalized = normalize_sql_query(gold_sql)
                        pred_normalized = normalize_sql_query(pred_sql)
                        sql_match = (gold_normalized == pred_normalized)
                        if sql_match:
                            results["overall_metrics"]["sql_correct"] += 1
                        sql_cont_res = {
                            'score': 1.0 if sql_match else 0.0,
                            'component_recall_score': 1.0 if sql_match else 0.0
                        }
                        exec_accuracy = None

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

            else:  # SQL evaluation
                # Get ground truth SQL
                gold_sql = item.get("SQL", item.get("sql", ""))
                db_id = item.get("db_id", "")

                # Parse predicted SQL
                pred_sql = self.parse_sql_output(pred_text)

                if pred_sql is None:
                    parse_error = True
                    error_message = "Failed to parse SQL from model output"
                    results["overall_metrics"]["parse_errors"] += 1
                    cont_res = {'score': 0, 'component_recall_score': 0}
                else:
                    # Evaluate SQL using Spider evaluation
                    try:
                        if kmaps is not None:
                            eval_result = evaluate_single_pair(
                                pred_sql=pred_sql,
                                gold_sql=gold_sql,
                                db_dir=database_path,
                                db_name=db_id,
                                etype='all',  # Evaluate both execution and matching
                                kmaps=kmaps
                            )

                            # Extract scores
                            is_correct = eval_result['exact'] == 1
                            exec_accuracy = eval_result['exec']

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
                        else:
                            # Evaluate without foreign key maps (works for BIRD or when tables.json unavailable)
                            try:
                                # Create empty kmaps dict for evaluation
                                empty_kmaps = {db_id: {}} if db_id else {}
                                eval_result = evaluate_single_pair(
                                    pred_sql=pred_sql,
                                    gold_sql=gold_sql,
                                    db_dir=database_path,
                                    db_name=db_id,
                                    etype='all',
                                    kmaps=empty_kmaps
                                )

                                # Extract scores
                                is_correct = eval_result['exact'] == 1
                                exec_accuracy = eval_result['exec']

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

                            except Exception as e:
                                # Fallback to simple string comparison
                                if debug:
                                    print(f"Warning: Full SQL evaluation failed, using string comparison: {e}")
                                gold_normalized = normalize_sql_query(gold_sql)
                                pred_normalized = normalize_sql_query(pred_sql)
                                is_correct = (gold_normalized == pred_normalized)
                                if is_correct:
                                    results["overall_metrics"]["correct"] += 1
                                cont_res = {
                                    'score': 1.0 if is_correct else 0.0,
                                    'component_recall_score': 1.0 if is_correct else 0.0
                                }
                    except Exception as e:
                        error_message = f"SQL evaluation error: {str(e)}"
                        cont_res = {'score': 0, 'component_recall_score': 0}
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
            else:  # SQL
                result_entry["ground_truth_sql"] = gold_sql
                result_entry["predicted_sql"] = pred_sql
                if exec_accuracy is not None:
                    result_entry["exec_accuracy"] = exec_accuracy

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
        if task_type in ['sql', 'ra_sql']:
            results["overall_metrics"]["exec_accuracy"] = (
                results["overall_metrics"]["exec_correct"] / results["overall_metrics"]["total"]
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
        print(f"EVALUATION SUMMARY (vLLM) - {task_type.upper()} Task")
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
                print(f"  Execution Accuracy: {metrics['exec_accuracy']*100:.2f}%")
                print(f"  Execution Correct: {metrics['exec_correct']}")
        else:
            print(f"Correct: {metrics['correct']}")
            print(f"Incorrect: {metrics['total'] - metrics['correct']}")
            print(f"Accuracy: {metrics['accuracy']*100:.2f}%")

            if task_type == 'sql' and 'exec_accuracy' in metrics:
                print(f"Execution Accuracy: {metrics['exec_accuracy']*100:.2f}%")
                print(f"Execution Correct: {metrics['exec_correct']}")

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
        description="Evaluate Relational Algebra or SQL SFT Model with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RA evaluation
  %(prog)s --model_checkpoint_path ./ra_model --input_file test_ra.json \\
           --database_path ./db --dataset_name spider \\
           --table_value_cache_path ./cache --table_info_cache_path ./info \\
           --task_type ra

  # SQL evaluation with Spider metrics (with foreign keys)
  %(prog)s --model_checkpoint_path ./sql_model --input_file test_sql.json \\
           --database_path ./db --dataset_name spider \\
           --table_value_cache_path ./cache --table_info_cache_path ./info \\
           --table_json_path ./tables.json --task_type sql

  # SQL evaluation for BIRD (no tables.json needed)
  %(prog)s --model_checkpoint_path ./sql_model --input_file bird_sql.json \\
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
           --tokenizer_name Qwen/Qwen2.5-0.5B \\
           --temp_dir_base ./vllm_temp ...
        """
    )
    
    # Model arguments
    parser.add_argument("--model_checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer/base model name if checkpoint lacks config (e.g., Qwen/Qwen2.5-0.5B)")
    
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
    parser.add_argument("--output_log", type=str, default="evaluation_results_vllm.json",
                        help="Path to save detailed evaluation results")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2],
                        help="Verbosity level: 0=summary only, 1=show first 10 and errors, 2=show all")
    parser.add_argument("--detailed_log", type=str, default=None,
                        help="Path to save detailed text log with all outputs")
    parser.add_argument("--save_errors", action="store_true",
                        help="Save errors to a separate JSON file for analysis")
    
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
    parser.add_argument('--task_type', type=str, default='ra', choices=['ra', 'sql', 'ra_sql'],
                        help='Task type: "ra" for relational algebra, "sql" for SQL evaluation, "ra_sql" for combined RA+SQL evaluation')
    parser.add_argument('--table_json_path', type=str, default=None,
                        help='Path to tables.json for SQL evaluation (optional, improves Spider evaluation accuracy)')

    args = parser.parse_args()

    # Warn if table_json_path not provided for SQL or RA+SQL evaluation
    if args.task_type in ['sql', 'ra_sql'] and not args.table_json_path:
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
    evaluator = RelationalAlgebraVLLMEvaluator(
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
    )
    
    # Run evaluation
    print(f"\nStarting {args.task_type.upper()} evaluation with vLLM...")
    print(f"Verbosity level: {args.verbose} (0=summary, 1=first 10 + errors, 2=all)")
    if args.detailed_log:
        print(f"Detailed log will be saved to: {args.detailed_log}")
    if args.task_type == 'sql':
        print(f"Using tables.json from: {args.table_json_path}")
    
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
        table_json_path=args.table_json_path
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
