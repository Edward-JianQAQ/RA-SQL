"""
Example usage (single GPU):
python ra_train.py \
--model_name_or_path=Qwen/Qwen2.5-3B \
--max_length=4096 \
--batch_size=4 \
--num_epochs=3 \
--database_path=ra_data/bird/train/train_databases \
--dataset_name=bird \
--table_value_cache_path=ra_data/bird/train/bird_train_id2sampled_values.json \
--table_info_cache_path=ra_data/bird/train/bird_train_id2db_info.json \
--input_file=cot_gen_res/new_data/bird_v1.json \
--use_wandb \
--task_type=ra \
--cot \
--serialize_cot_type=post_order

Example usage (multinode with DeepSpeed):
deepspeed --hostfile=host_file --no_ssh --node_rank=0 --master_addr=snorlax-2 --master_port=12345 ra_train.py \
--model_name_or_path=Qwen/Qwen2.5-3B \
--max_length=4096 \
--batch_size=4 \
--num_epochs=3 \
--database_path=ra_data/bird/train/train_databases \
--dataset_name=bird \
--table_value_cache_path=ra_data/bird/train/bird_train_id2sampled_values.json \
--table_info_cache_path=ra_data/bird/train/bird_train_id2db_info.json \
--input_file=cot_gen_res/new_data/bird_v1.json \
--use_wandb \
--task_type=ra \
--cot \
--serialize_cot_type=post_order \
--deepspeed \
--deepspeed_config=ds_config_zero3.json
"""

import argparse
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datetime import datetime
from data_utils import get_input_seq_ra, get_input_seq as get_input_seq_sql, get_input_seq_ra_sql
from training_prompts import (
    format_ra_answer,
    format_sql_answer,
    format_ra_sql_answer,
    extract_json_from_text,
    extract_sql_from_text,
    extract_thinking_content,
    get_sql_field_from_item
)
from serialization.ra_serial import (
    resolve_ra_pointers,
    serialize_post_order_narrative,
    serialize_pre_order_summary,
    serialize_indented_narrative,
    serialize_random_plan_preorder,
)



import sqlparse

import torch
from torch.utils.data import Dataset


def normalize_sql_query(sql):
    """Normalize a SQL query for comparison."""
    if not sql:
        return ""
    return sqlparse.format(sql, strip_comments=True, reindent=False, keyword_case="upper").strip().rstrip(";").strip()
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import wandb

class RelationalQueryDataset(Dataset):
    """Dataset for relational query SFT training (supports both RA and SQL).
    
    Each sample contains a prompt and expected output (RA in JSON format or SQL query).
    """
    
    def __init__(
        self,
        input_dataset,
        database_path: str,
        dataset_name: str,
        table_value_cache_path: str,
        table_info_cache_path: str,
        tokenizer: AutoTokenizer,
        task_type: str = 'ra',  # 'ra' or 'sql'
        max_length: int = 2048,
        mode: str = 'train',
        debug_max_items: Optional[int] = None,
        cot: bool = True,
        serialize_cot_type: str = "",
        full_loss: bool = False,
        seed: Optional[int] = None,
        save_prompt_examples: int = 0,  # Number of prompt examples to save
    ) -> None:
        self.samples: List[Dict[str, Any]] = []
        self.prompt_examples: List[Dict[str, str]] = []  # Store raw text examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.seed = seed
        self.save_prompt_examples = save_prompt_examples
        
        if task_type not in ['ra', 'sql', 'ra_sql']:
            raise ValueError(f"task_type must be 'ra', 'sql', or 'ra_sql', got {task_type}")
        
        # Process dataset
        for idx, item in enumerate(tqdm(input_dataset, desc=f"Processing {mode} {task_type} data")):
            if debug_max_items is not None and idx >= debug_max_items:
                break
                
            # Get prompt based on task type
            if self.task_type == 'ra':
                prompt = get_input_seq_ra(
                    item['item'],
                    database_path,
                    dataset_name,
                    table_value_cache_path,
                    table_info_cache_path,
                    mode=mode,
                    cot=cot
                )

                # pdb.set_trace()
                
                # Get relational algebra ground truth
                ra_dict = item['item']["relational_algebra"]

                # Extract CoT content if available
                cot_content = None
                if cot:
                    if serialize_cot_type:  # If not empty string, use ra_serial
                        # Generate CoT from RA tree using ra_serial
                        resolved_tree = resolve_ra_pointers(ra_dict, annotate_schema=True)

                        # Select serialization method based on type
                        if serialize_cot_type == "post_order":
                            cot_content = serialize_post_order_narrative(resolved_tree)
                        elif serialize_cot_type == "pre_order":
                            cot_content = serialize_pre_order_summary(resolved_tree)
                        elif serialize_cot_type == "indented":
                            cot_content = serialize_indented_narrative(resolved_tree)
                        elif serialize_cot_type == "random_preorder":
                            if self.seed is not None:
                                self.seed += 1
                            cot_content = serialize_random_plan_preorder(resolved_tree, seed=self.seed)

                    elif 'llm_response' in item:
                        cot_content = extract_thinking_content(item['llm_response'])

                # Format answer using the new system
                answer = format_ra_answer(ra_dict, cot_content, tokenizer.eos_token)
                
            elif self.task_type == 'ra_sql':
                prompt = get_input_seq_ra_sql(
                    item['item'],
                    database_path,
                    dataset_name,
                    table_value_cache_path,
                    table_info_cache_path,
                    mode=mode,
                    cot=cot
                )
                
                # Get both RA and SQL ground truth
                ra_dict = item['item']["relational_algebra"]
                sql_query = get_sql_field_from_item(item['item'], dataset_name)

                # Extract CoT content if available
                cot_content = None
                if cot:
                    if serialize_cot_type:  # If not empty string, use ra_serial
                        # Generate CoT from RA tree using ra_serial
                        resolved_tree = resolve_ra_pointers(ra_dict, annotate_schema=True)

                        # Select serialization method based on type
                        if serialize_cot_type == "post_order":
                            cot_content = serialize_post_order_narrative(resolved_tree)
                        elif serialize_cot_type == "pre_order":
                            cot_content = serialize_pre_order_summary(resolved_tree)
                        elif serialize_cot_type == "indented":
                            cot_content = serialize_indented_narrative(resolved_tree)
                    elif 'llm_response' in item:
                        cot_content = extract_thinking_content(item['llm_response'])

                # Format answer using the new system
                answer = format_ra_sql_answer(ra_dict, sql_query, cot_content, tokenizer.eos_token)
                
            elif self.task_type == 'sql':
                prompt = get_input_seq_sql(
                    item['item'],
                    database_path,
                    dataset_name,
                    table_value_cache_path,
                    table_info_cache_path,
                    mode=mode,
                    cot=cot  # Pass for future compatibility, even if not used yet
                )
                
                # Get SQL ground truth
                sql_query = get_sql_field_from_item(item['item'], dataset_name)

                # Extract CoT content if available
                cot_content = None
                if cot:
                    if serialize_cot_type:  # If not empty string, use ra_serial
                        # For SQL task, generate CoT from RA tree if available
                        if "relational_algebra" in item['item']:
                            ra_dict = item['item']["relational_algebra"]
                            resolved_tree = resolve_ra_pointers(ra_dict, annotate_schema=True)

                            # Select serialization method based on type
                            if serialize_cot_type == "post_order":
                                cot_content = serialize_post_order_narrative(resolved_tree)
                            elif serialize_cot_type == "pre_order":
                                cot_content = serialize_pre_order_summary(resolved_tree)
                            elif serialize_cot_type == "indented":
                                cot_content = serialize_indented_narrative(resolved_tree)
                    elif 'llm_response' in item:
                        cot_content = extract_thinking_content(item['llm_response'])

                # Format answer using the new system
                answer = format_sql_answer(sql_query, cot_content, tokenizer.eos_token)
            
            # Tokenize prompt and answer separately (no special tokens)
            prompt_enc = tokenizer(prompt + "\n", add_special_tokens=False)
            answer_enc = tokenizer(answer, add_special_tokens=False)
            
            # Combine tokens
            input_ids = prompt_enc["input_ids"] + answer_enc["input_ids"]
            attention_mask = [1] * len(input_ids)
            
            # Create labels - mask prompt tokens with -100 unless full_loss is True
            if not full_loss:
                labels = ([-100] * len(prompt_enc["input_ids"]) + answer_enc["input_ids"])
            else:
                labels = prompt_enc["input_ids"] + answer_enc["input_ids"]
            
            # Truncate if needed
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
            
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

            # Save prompt examples if requested
            if len(self.prompt_examples) < self.save_prompt_examples:
                self.prompt_examples.append({
                    "prompt": prompt,
                    "answer": answer,
                    "full_text": prompt + "\n" + answer,
                    "task_type": self.task_type,
                    "mode": mode,
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]



class DataCollatorForCausalLMMasking:
    """Data collator that handles padding and masking for causal LM."""
    
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]
        attn_list = [f["attention_mask"] for f in features]
        
        batch_input = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attn_list},
            padding=True,
            return_tensors="pt",
        )
        max_seq_len = batch_input["input_ids"].size(1)
        
        padded_labels = []
        for lbl in labels_list:
            lbl = lbl + [-100] * (max_seq_len - len(lbl))
            padded_labels.append(lbl)
        batch_labels = torch.tensor(padded_labels, dtype=torch.long)
        
        return {
            "input_ids": batch_input["input_ids"],
            "attention_mask": batch_input["attention_mask"],
            "labels": batch_labels,
        }


@torch.no_grad()
def evaluate_ra_accuracy(
    model,
    tokenizer,
    val_dataset,
    database_path: str,
    dataset_name: str,
    table_value_cache_path: str,
    table_info_cache_path: str,
    max_new_tokens: int = 512,
    sample_size: int = 100,
    cot: bool = True
) -> float:
    """Evaluate exact match accuracy for RA prediction."""
    
    model.eval()
    total, correct = 0, 0
    
    # Sample subset for evaluation
    eval_items = val_dataset[:min(sample_size, len(val_dataset))]
    
    for item in tqdm(eval_items, desc="Evaluating"):
        total += 1
        
        # Get prompt
        prompt = get_input_seq_ra(
            item['item'],
            database_path,
            dataset_name,
            table_value_cache_path,
            table_info_cache_path,
            mode='dev',
            cot=cot
        )
        
        # Get ground truth
        gold_ra = item['item']["relational_algebra"]
        
        # Tokenize prompt
        enc = tokenizer(prompt + "\n", return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        # Generate (greedy decoding)
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0][enc["input_ids"].size(1):]
        
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        # Extract JSON from prediction
        try:
            pred_ra = extract_json_from_text(pred_text)
            if pred_ra and pred_ra == gold_ra:
                correct += 1
        except:
            pass  # Failed to parse, counts as incorrect
    
    return correct / total if total else 0.0


@torch.no_grad()
def evaluate_query_accuracy(
    model,
    tokenizer,
    val_dataset,
    database_path: str,
    dataset_name: str,
    table_value_cache_path: str,
    table_info_cache_path: str,
    task_type: str = 'ra',  # 'ra' or 'sql'
    max_new_tokens: int = 512,
    sample_size: int = 100,
    cot: bool = True,
    normalize_sql: bool = True
) -> float:
    """Evaluate exact match accuracy for query prediction (RA or SQL)."""
    
    if task_type not in ['ra', 'sql']:
        raise ValueError(f"task_type must be 'ra' or 'sql', got {task_type}")
    
    model.eval()
    total, correct = 0, 0
    
    # Sample subset for evaluation
    eval_items = val_dataset[:min(sample_size, len(val_dataset))]
    
    for item in tqdm(eval_items, desc=f"Evaluating {task_type.upper()}"):
        total += 1
        
        # Get prompt based on task type
        if task_type == 'ra':
            prompt = get_input_seq_ra(
                item['item'],
                database_path,
                dataset_name,
                table_value_cache_path,
                table_info_cache_path,
                mode='dev',
                cot=cot
            )
            gold_answer = item['item']["relational_algebra"]
        else:  # sql
            prompt = get_input_seq_sql(
                item['item'],
                database_path,
                dataset_name,
                table_value_cache_path,
                table_info_cache_path,
                mode='dev',
                cot=cot
            )
            gold_answer = item['item']['SQL']
        
        # Tokenize prompt
        enc = tokenizer(prompt + "\n", return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        # Generate (greedy decoding)
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0][enc["input_ids"].size(1):]
        
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        # Extract and compare based on task type
        try:
            if task_type == 'ra':
                pred_answer = extract_json_from_text(pred_text)
                if pred_answer and pred_answer == gold_answer:
                    correct += 1
            else:  # sql
                pred_answer = extract_sql_from_text(pred_text)
                if pred_answer:
                    # Normalize and compare SQL
                    if normalize_sql:
                        pred_normalized = normalize_sql_query(pred_answer)
                        gold_normalized = normalize_sql_query(gold_answer)
                        if pred_normalized == gold_normalized:
                            correct += 1
                    else:
                        if pred_answer == gold_answer:
                            correct += 1
        except:
            pass  # Failed to parse, counts as incorrect
    
    return correct / total if total else 0.0


@torch.no_grad()
def evaluate_ra_sql_accuracy(
    model,
    tokenizer,
    val_dataset,
    database_path: str,
    dataset_name: str,
    table_value_cache_path: str,
    table_info_cache_path: str,
    max_new_tokens: int = 1024,  # Increased for both RA and SQL
    sample_size: int = 100,
    cot: bool = True,
    normalize_sql: bool = True
) -> dict:
    """Evaluate accuracy for RA+SQL mode where both are generated."""
    
    model.eval()
    total = 0
    ra_correct = 0
    sql_correct = 0
    both_correct = 0
    
    # Sample subset for evaluation
    eval_items = val_dataset[:min(sample_size, len(val_dataset))]
    
    for item in tqdm(eval_items, desc="Evaluating RA+SQL"):
        total += 1
        
        # Get prompt
        prompt = get_input_seq_ra_sql(
            item['item'],
            database_path,
            dataset_name,
            table_value_cache_path,
            table_info_cache_path,
            mode='dev',
            cot=cot
        )
        
        # Get ground truth
        gold_ra = item['item']["relational_algebra"]
        gold_sql = get_sql_field_from_item(item['item'], dataset_name)
        
        # Tokenize prompt
        enc = tokenizer(prompt + "\n", return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        
        # Generate (greedy decoding)
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0][enc["input_ids"].size(1):]
        
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        # Extract RA from <think> section
        ra_match = False
        sql_match = False
        
        try:
            # Extract RA from thinking section
            think_content = extract_thinking_content(pred_text)
            if think_content:
                pred_ra = extract_json_from_text(think_content)
                if pred_ra and pred_ra == gold_ra:
                    ra_match = True
                    ra_correct += 1
            else:
                # Try whole text if no thinking section
                pred_ra = extract_json_from_text(pred_text)
                if pred_ra and pred_ra == gold_ra:
                    ra_match = True
                    ra_correct += 1
        except:
            pass  # Failed to parse RA
        
        try:
            # Extract SQL from answer section or whole text
            pred_sql = extract_sql_from_text(pred_text)
            if pred_sql:
                # Normalize and compare SQL
                if normalize_sql:
                    pred_normalized = normalize_sql_query(pred_sql)
                    gold_normalized = normalize_sql_query(gold_sql)
                    if pred_normalized == gold_normalized:
                        sql_match = True
                        sql_correct += 1
                else:
                    if pred_sql == gold_sql:
                        sql_match = True
                        sql_correct += 1
        except:
            pass  # Failed to parse SQL
        
        if ra_match and sql_match:
            both_correct += 1
    
    return {
        "ra_accuracy": ra_correct / total if total else 0.0,
        "sql_accuracy": sql_correct / total if total else 0.0,
        "both_accuracy": both_correct / total if total else 0.0,
        "total": total
    }


def main():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--table_value_cache_path", type=str, required=True)
    parser.add_argument("--table_info_cache_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    
    # Model/tokenizer
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output_dir", type=str, default="sft_model")
    
    # Training hyperparameters
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    
    # Misc
    parser.add_argument("--debug_dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--serialize_cot_type", type=str, default="",
                        choices=["", "post_order", "pre_order", "indented", "random_preorder"],
                        help="CoT serialization method. **RECOMMENDED V4** (semantic reasoning): hybrid_reasoning_v4 (best - WHY+WHAT), detailed_reasoning_v4 (more detail). V3 (structure): hybrid_v3, step_plan_v3. Original: pre_order, post_order.")
    parser.add_argument("--full_loss", action="store_true")
    parser.add_argument('--task_type', type=str, choices=['ra', 'sql', 'ra_sql'], default='ra')
    parser.add_argument("--num_prompt_examples", type=int, default=5, help="Number of prompt examples to log during training")

    # Distributed training / DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for distributed training")
    parser.add_argument("--deepspeed", action="store_true", help="enable DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config_zero3.json", help="path to DeepSpeed config file")
    parser.add_argument("--save_limit", type=int, default=3, help="number of checkpoints to keep")
    parser.add_argument("--sel_seed", type=int, default=42, help="random seed for evaluation sampling")
    args = parser.parse_args()

    import os

    torch.manual_seed(args.seed)

    # Setup for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)

    # Extract model name from path (e.g., "Qwen/Qwen2.5-7B" -> "Qwen2.5-7B")
    model_name = args.model_name_or_path.split('/')[-1]

    # Construct output directory with model name under ra_sql_ckpts
    checkpoint_dir = f'{args.task_type}_{model_name}_' + args.output_dir

    if args.cot:
        checkpoint_dir += "_cot"
        if args.serialize_cot_type:
            checkpoint_dir += f"_{args.serialize_cot_type}"

    if args.full_loss:
        checkpoint_dir += "_full_loss"

    # Place under ra_sql_ckpts parent folder
    args.output_dir = os.path.join("ra_sql_ckpts", checkpoint_dir)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save hyperparameters to file
    hyperparams_path = os.path.join(args.output_dir, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Hyperparameters saved to: {hyperparams_path}")

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="relational-algebra-sft",
            config=vars(args),
            save_code=True  # This saves the main script
        )
        
        # Save wandb run URL to file in output directory
        wandb_url = wandb.run.get_url() if wandb.run else None
        if wandb_url:
            wandb_info_path = os.path.join(args.output_dir, "wandb_run_info.txt")
            with open(wandb_info_path, "w") as f:
                f.write(f"WandB Run URL: {wandb_url}\n")
                f.write(f"WandB Run ID: {wandb.run.id}\n")
                f.write(f"WandB Run Name: {wandb.run.name}\n")
                f.write(f"WandB Project: {wandb.run.project}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            print(f"WandB run info saved to: {wandb_info_path}")
            print(f"WandB run URL: {wandb_url}")
        
        # Save additional Python files to wandb
        wandb.save("*.py", policy="now")  # Save all Python files in current directory
        wandb.save("data_utils.py", policy="now")
        wandb.save("training_prompts.py", policy="now")
        wandb.save("prompt_manager.py", policy="now")
        
        # Save all prompt templates
        wandb.save("prompt_templates/**/*.txt", policy="now")  # Save all txt files in prompt_templates
        wandb.save("prompt_templates/README.md", policy="now")  # Save the README
        
        # Optionally save config files if you have any
        # wandb.save("*.json", policy="now")
        # wandb.save("*.yaml", policy="now")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # Load and split dataset
    input_dataset = json.load(open(args.input_file))
    # Normalize format: wrap flat items in {"item": ...} if needed
    if input_dataset and "item" not in input_dataset[0]:
        input_dataset = [{"item": item} for item in input_dataset]
    
    # Split into train/val
    split_idx = int(len(input_dataset) * args.train_split)
    train_data = input_dataset[:split_idx]
    val_data = input_dataset[split_idx:]
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = RelationalQueryDataset(
        input_dataset=train_data,
        database_path=args.database_path,
        dataset_name=args.dataset_name,
        table_value_cache_path=args.table_value_cache_path,
        table_info_cache_path=args.table_info_cache_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode='train',
        debug_max_items=32 if args.debug_dataset else None,
        cot=args.cot,
        serialize_cot_type=args.serialize_cot_type,
        full_loss=args.full_loss,
        task_type=args.task_type,
        seed=args.sel_seed,
        save_prompt_examples=args.num_prompt_examples
    )

    # Save prompt examples to file (only on rank 0 in distributed setting)
    if rank == 0 and args.num_prompt_examples > 0 and len(train_dataset.prompt_examples) > 0:
        prompt_examples_path = os.path.join(args.output_dir, "prompt_examples.json")
        with open(prompt_examples_path, "w") as f:
            json.dump(train_dataset.prompt_examples, f, indent=2)
        print(f"Saved {len(train_dataset.prompt_examples)} prompt examples to: {prompt_examples_path}")
    
    val_dataset = RelationalQueryDataset(
        input_dataset=val_data,
        database_path=args.database_path,
        dataset_name=args.dataset_name,
        table_value_cache_path=args.table_value_cache_path,
        table_info_cache_path=args.table_info_cache_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mode='dev',
        debug_max_items=16 if args.debug_dataset else None,
        cot=args.cot,
        serialize_cot_type=args.serialize_cot_type,
        full_loss=args.full_loss,
        task_type=args.task_type,
        seed=args.sel_seed
    )
    
    # Data collator
    collator = DataCollatorForCausalLMMasking(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,  # Save at same frequency as eval
        eval_steps=args.eval_steps,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        save_total_limit=args.save_limit,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        report_to=["wandb"] if args.use_wandb else [],
        deepspeed=args.deepspeed_config if args.deepspeed else None,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Save padded prompt examples (showing actual batched/padded sequences)
    if rank == 0 and args.num_prompt_examples > 0 and len(train_dataset) > 0:
        # Get first few samples and collate them to see padding
        num_samples_to_log = min(args.num_prompt_examples, len(train_dataset))
        sample_batch = [train_dataset[i] for i in range(num_samples_to_log)]
        collated_batch = collator(sample_batch)

        padded_examples = []
        for i in range(num_samples_to_log):
            input_ids = collated_batch["input_ids"][i].tolist()
            labels = collated_batch["labels"][i].tolist()
            attention_mask = collated_batch["attention_mask"][i].tolist()

            # Decode with special tokens to see padding
            padded_text = tokenizer.decode(input_ids, skip_special_tokens=False)

            # Also show where the labels are masked
            label_mask_positions = [j for j, lbl in enumerate(labels) if lbl == -100]

            padded_examples.append({
                "index": i,
                "padded_text": padded_text,
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "label_mask_positions": label_mask_positions,
                "sequence_length": len(input_ids),
                "num_padding_tokens": attention_mask.count(0),
                "padding_side": tokenizer.padding_side if hasattr(tokenizer, 'padding_side') else "right",
            })

        padded_examples_path = os.path.join(args.output_dir, "padded_prompt_examples.json")
        with open(padded_examples_path, "w") as f:
            json.dump(padded_examples, f, indent=2)
        print(f"Saved {len(padded_examples)} padded prompt examples to: {padded_examples_path}")

    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    # Evaluate
    if val_data:
        if args.task_type == 'ra_sql':
            # Evaluate both RA and SQL
            results = evaluate_ra_sql_accuracy(
                model,
                tokenizer,
                val_data,
                args.database_path,
                args.dataset_name,
                args.table_value_cache_path,
                args.table_info_cache_path,
                sample_size=50,
                cot=args.cot,
                max_new_tokens=args.max_length,
                normalize_sql=True
            )
            print(f"RA Exact-match accuracy: {results['ra_accuracy'] * 100:.2f}%")
            print(f"SQL Exact-match accuracy: {results['sql_accuracy'] * 100:.2f}%")
            print(f"Both correct: {results['both_accuracy'] * 100:.2f}%")
            
            if args.use_wandb:
                wandb.log({
                    "final_ra_accuracy": results['ra_accuracy'],
                    "final_sql_accuracy": results['sql_accuracy'],
                    "final_both_accuracy": results['both_accuracy']
                })
        else:
            # Use existing evaluation for ra or sql
            accuracy = evaluate_query_accuracy(
                model,
                tokenizer,
                val_data,
                args.database_path,
                args.dataset_name,
                args.table_value_cache_path,
                args.table_info_cache_path,
                sample_size=50,
                cot=args.cot,
                max_new_tokens=args.max_length,
                task_type=args.task_type,
                normalize_sql=True if args.task_type == 'sql' else False
            )
            print(f"Exact-match accuracy on validation: {accuracy * 100:.2f}%")
            
            if args.use_wandb:
                wandb.log({"final_accuracy": accuracy})
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()