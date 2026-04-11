"""
Chat-aligned SFT training script for RA/SQL tasks.

Uses chat template (tokenizer.apply_chat_template on [system, user] + assistant answer).

This mirrors the data flow of ra_train.py but wraps prompts/answers as chat messages.
"""

import argparse
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import wandb
from tqdm import tqdm

from data_utils import (
    get_input_seq_ra,
    get_input_seq as get_input_seq_sql,
    get_input_seq_ra_sql,
)
from training_prompts import (
    format_ra_answer,
    format_sql_answer,
    format_ra_sql_answer,
    extract_thinking_content,
)
from serialization.ra_serial import (
    resolve_ra_pointers,
    serialize_post_order_narrative,
    serialize_pre_order_summary,
    serialize_indented_narrative,
    serialize_random_plan_preorder,
)




DEFAULT_SYSTEM_MESSAGE = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)


class ChatAlignedRelationalQueryDataset(Dataset):
    """Dataset for chat-aligned SFT (supports RA, SQL, RA+SQL, and CoT).

    For each sample, we build a chat prompt using Qwen's chat template:
    messages = [
        {role: system, content: system_message},
        {role: user,   content: prompt_text},
    ]
    and then append the assistant answer (RA/SQL/CoT string with tags) as target.

    Task types:
    - 'ra': Generate relational algebra
    - 'sql': Generate SQL directly
    - 'ra_sql': Generate both RA and SQL
    - 'cot': Generate chain-of-thought reasoning (SynSQL-style)
    """

    def __init__(
        self,
        input_dataset,
        database_path: str,
        dataset_name: str,
        table_value_cache_path: str,
        table_info_cache_path: str,
        tokenizer: AutoTokenizer,
        task_type: str = 'ra',  # 'ra', 'sql', or 'ra_sql'
        max_length: int = 4096,
        mode: str = 'train',
        debug_max_items: Optional[int] = None,
        cot: bool = False,
        serialize_cot_type: str = "",
        full_loss: bool = False,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        sel_seed: Optional[int] = None,
        save_prompt_examples: int = 0,  # Number of prompt examples to save,
        apply_template: bool = False,
        # Multi-dataset support
        combined_datasets: Optional[List[Dict[str, Any]]] = None,
    ):
        self.samples: List[Dict[str, Any]] = []
        self.prompt_examples: List[Dict[str, str]] = []  # Store raw text examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.full_loss = full_loss
        self.system_message = system_message
        self.sel_seed = sel_seed
        self.save_prompt_examples = save_prompt_examples

        # Support combining multiple datasets
        if combined_datasets is not None:
            # combined_datasets is a list of dicts with keys:
            # 'input_dataset', 'database_path', 'dataset_name', 'table_value_cache_path', 'table_info_cache_path'
            all_items = []
            for ds_config in combined_datasets:
                items = ds_config['input_dataset']
                # Tag each item with its dataset config
                for item in items:
                    tagged_item = {
                        'item': item["item"] if isinstance(item, dict) and "item" in item else item,
                        'database_path': ds_config['database_path'],
                        'dataset_name': ds_config['dataset_name'],
                        'table_value_cache_path': ds_config['table_value_cache_path'],
                        'table_info_cache_path': ds_config['table_info_cache_path'],
                    }
                    all_items.append(tagged_item)

            if debug_max_items is not None:
                all_items = all_items[:debug_max_items]

            print(f"Building chat-aligned dataset with {len(combined_datasets)} combined datasets, total {len(all_items)} items")
            self._process_items(all_items, mode, cot, serialize_cot_type, apply_template, use_tagged=True)
        else:
            # Single dataset mode (original behavior)
            items = input_dataset
            if debug_max_items is not None:
                items = items[:debug_max_items]

            print(f"Building chat-aligned dataset for {len(items)} items, task_type={self.task_type}, cot={cot}, serialize_cot_type={serialize_cot_type}, apply_template={apply_template}")

            # Tag items with single dataset config
            tagged_items = []
            for item in items:
                tagged_item = {
                    'item': item["item"] if isinstance(item, dict) and "item" in item else item,
                    'database_path': database_path,
                    'dataset_name': dataset_name,
                    'table_value_cache_path': table_value_cache_path,
                    'table_info_cache_path': table_info_cache_path,
                }
                tagged_items.append(tagged_item)

            self._process_items(tagged_items, mode, cot, serialize_cot_type, apply_template, use_tagged=True)

    def _process_items(self, items, mode, cot, serialize_cot_type, apply_template, use_tagged=False):
        """Process items and build samples."""
        for item_data in tqdm(items, desc="Building dataset", unit="sample"):
            if use_tagged:
                eg = item_data['item']
                database_path = item_data['database_path']
                dataset_name = item_data['dataset_name']
                table_value_cache_path = item_data['table_value_cache_path']
                table_info_cache_path = item_data['table_info_cache_path']
            else:
                # Legacy path (not used with new structure)
                eg = item_data["item"] if isinstance(item_data, dict) and "item" in item_data else item_data
                database_path = self.database_path
                dataset_name = self.dataset_name
                table_value_cache_path = self.table_value_cache_path
                table_info_cache_path = self.table_info_cache_path

            # 1) Build prompt text
            if self.task_type == 'ra':
                prompt_text = get_input_seq_ra(
                    eg, database_path, dataset_name,
                    table_value_cache_path, table_info_cache_path,
                    mode=mode, cot=cot
                )
            elif self.task_type == 'ra_sql':
                prompt_text = get_input_seq_ra_sql(
                    eg, database_path, dataset_name,
                    table_value_cache_path, table_info_cache_path,
                    mode=mode, cot=cot
                )
            elif self.task_type == 'cot':
                # CoT task: use SQL-style prompt (question + schema)
                prompt_text = get_input_seq_sql(
                    eg, database_path, dataset_name,
                    table_value_cache_path, table_info_cache_path,
                    mode=mode, cot=False  # Don't add CoT in prompt
                )
            else:  # 'sql'
                prompt_text = get_input_seq_sql(
                    eg, database_path, dataset_name,
                    table_value_cache_path, table_info_cache_path,
                    mode=mode, cot=cot
                )

            # 2) Build assistant answer string
            if self.task_type == 'ra':
                if "relational_algebra" not in eg:
                    print(f"\n!!! ERROR: Item missing 'relational_algebra' field !!!")
                    print(f"Dataset: {dataset_name}")
                    print(f"Available keys: {list(eg.keys())}")
                    if "question" in eg:
                        print(f"Question: {eg['question'][:100]}...")
                    raise KeyError(f"'relational_algebra' field not found in item from dataset '{dataset_name}'. "
                                 f"Available keys: {list(eg.keys())}. "
                                 f"Make sure you're using data with RA annotations (e.g., cot_gen_res/new_data/spider_v1.json) "
                                 f"not plain SQL data (e.g., ra_data/spider/train_spider.json)")
                ra_dict = eg["relational_algebra"]
                cot_content = None
                if cot:
                    if serialize_cot_type:
                        resolved_tree = resolve_ra_pointers(ra_dict, annotate_schema=True)
                        if serialize_cot_type == "post_order":
                            cot_content = serialize_post_order_narrative(resolved_tree)
                        elif serialize_cot_type == "pre_order":
                            cot_content = serialize_pre_order_summary(resolved_tree)
                        elif serialize_cot_type == "indented":
                            cot_content = serialize_indented_narrative(resolved_tree)
                        elif serialize_cot_type == "random_preorder":
                            if self.sel_seed is not None:
                                self.sel_seed += 1
                            cot_content = serialize_random_plan_preorder(resolved_tree, seed=self.sel_seed)
                                                # V2 methods (less effective)
                        elif serialize_cot_type == "execution_plan_v2":
                            cot_content = serialize_recommended(
                                resolved_tree,
                                format_type="execution_plan",
                                seed=self.sel_seed
                            )
                        elif serialize_cot_type == "minimal_compact_v2":
                            cot_content = serialize_recommended(
                                resolved_tree,
                                format_type="minimal_compact",
                                seed=self.sel_seed
                            )
                        # V3 methods (analysis-driven, structure-focused)
                        elif serialize_cot_type == "compact_tree_v3":
                            cot_content = serialize_compact_tree_v3(resolved_tree)
                        elif serialize_cot_type == "step_plan_v3":
                            cot_content = serialize_step_plan_v3(resolved_tree)
                        elif serialize_cot_type == "focused_pre_v3":
                            cot_content = serialize_focused_preorder_v3(resolved_tree)
                        elif serialize_cot_type == "hybrid_v3":
                            if self.sel_seed is not None:
                                self.sel_seed += 1
                            cot_content = serialize_hybrid_compact_v3(resolved_tree, seed=self.sel_seed)
                        # V4 methods (RECOMMENDED - semantic reasoning + structure)
                        elif serialize_cot_type == "hybrid_reasoning_v4":
                            cot_content = serialize_hybrid_reasoning_v4(resolved_tree)
                        elif serialize_cot_type == "detailed_reasoning_v4":
                            cot_content = serialize_detailed_reasoning_v4(resolved_tree)
                        elif serialize_cot_type == "compact_v4":
                            cot_content = serialize_compact_reasoning_v4(resolved_tree)

                    elif 'llm_response' in eg:
                        cot_content = extract_thinking_content(eg['llm_response'])
                answer_text = format_ra_answer(ra_dict, cot_content, self.tokenizer.eos_token, no_eos=apply_template)

            elif self.task_type == 'ra_sql':
                if "relational_algebra" not in eg:
                    print(f"\n!!! ERROR: Item missing 'relational_algebra' field !!!")
                    print(f"Dataset: {dataset_name}")
                    print(f"Available keys: {list(eg.keys())}")
                    if "question" in eg:
                        print(f"Question: {eg['question'][:100]}...")
                    raise KeyError(f"'relational_algebra' field not found in item from dataset '{dataset_name}'. "
                                 f"Available keys: {list(eg.keys())}. "
                                 f"For task_type='ra_sql', use data with RA annotations (e.g., cot_gen_res/new_data/spider_v1.json) "
                                 f"not plain SQL data (e.g., ra_data/spider/train_spider.json)")
                ra_dict = eg["relational_algebra"]
                from training_prompts import get_sql_field_from_item
                sql_query = get_sql_field_from_item(eg, dataset_name)
                cot_content = None
                if cot and serialize_cot_type:
                    resolved_tree = resolve_ra_pointers(ra_dict, annotate_schema=True)
                    if serialize_cot_type == "post_order":
                        cot_content = serialize_post_order_narrative(resolved_tree)
                    elif serialize_cot_type == "pre_order":
                        cot_content = serialize_pre_order_summary(resolved_tree)
                    elif serialize_cot_type == "indented":
                        cot_content = serialize_indented_narrative(resolved_tree)
                elif cot and 'llm_response' in eg:
                    cot_content = extract_thinking_content(eg['llm_response'])
                answer_text = format_ra_sql_answer(ra_dict, sql_query, cot_content, self.tokenizer.eos_token, no_eos=apply_template)

            elif self.task_type == 'cot':
                # CoT task: use the 'cot' field directly from data
                if "cot" not in eg:
                    print(f"\n!!! ERROR: Item missing 'cot' field !!!")
                    print(f"Dataset: {dataset_name}")
                    print(f"Available keys: {list(eg.keys())}")
                    if "question" in eg:
                        print(f"Question: {eg['question'][:100]}...")
                    raise KeyError(f"'cot' field not found in item from dataset '{dataset_name}'. "
                                 f"Available keys: {list(eg.keys())}. "
                                 f"For task_type='cot', use data with CoT annotations (e.g., SynSQL data with 'cot' field)")

                cot_content = eg["cot"]

                # Format answer: CoT reasoning followed by SQL
                # Match SynSQL format: just output the CoT content directly
                if apply_template:
                    answer_text = cot_content  # No EOS when using template
                else:
                    answer_text = cot_content + self.tokenizer.eos_token

            else:  # 'sql'
                from training_prompts import get_sql_field_from_item
                sql_query = get_sql_field_from_item(eg, dataset_name)
                answer_text = format_sql_answer(sql_query, cot_content=None, eos_token=self.tokenizer.eos_token, no_eos=apply_template)

            # 3) Build chat-formatted prompt (system + user), add generation prompt
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            messages.append({"role": "user", "content": prompt_text})

            if apply_template:
                # Correct approach: include assistant message in template
                messages.append({"role": "assistant", "content": answer_text})

                full_formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False  # False because we have the answer
                )

                # Tokenize the complete formatted conversation
                input_ids = self.tokenizer(full_formatted, add_special_tokens=False)["input_ids"]
                attention_mask = [1] * len(input_ids)

                if self.full_loss:
                    labels = input_ids.copy()
                else:
                    # Need to find where the assistant's response starts
                    # Build prompt-only version to find the boundary
                    prompt_chat = self.tokenizer.apply_chat_template(
                        messages[:-1],  # Without assistant message
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompt_ids = self.tokenizer(prompt_chat, add_special_tokens=False)["input_ids"]

                    # Mask prompt tokens, keep answer tokens
                    labels = ([-100] * len(prompt_ids)) + input_ids[len(prompt_ids):]
            else:

                prompt_chat = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # 4) Tokenize prompt and answer separately
                prompt_ids = self.tokenizer(prompt_chat, add_special_tokens=False)["input_ids"]
                answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]

                input_ids = prompt_ids + answer_ids
                attention_mask = [1] * len(input_ids)

                if self.full_loss:
                    labels = input_ids.copy()
                else:
                    labels = ([-100] * len(prompt_ids)) + answer_ids

            # print when truncation happens
            if len(input_ids) >= self.max_length:
                print(f"Truncation occurred for item with input length {len(input_ids)}")
            
            # Truncate (keep prefix as in ra_train.py)
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
                    "system_message": self.system_message,
                    "prompt": prompt_text,
                    "answer": answer_text,
                    "chat_prompt": prompt_chat,
                    "full_text": prompt_chat + answer_text,
                    "task_type": self.task_type,
                    "mode": mode,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class DataCollatorForCausalLMMasking:
    """Data collator that handles padding and masking for causal LM.

    Respects tokenizer.padding_side (supports both left and right padding).
    """

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

        # Pad labels according to tokenizer's padding_side
        padded_labels = []
        for lbl in labels_list:
            padding_length = max_seq_len - len(lbl)
            if self.tokenizer.padding_side == "left":
                # Left padding: prepend -100 tokens
                padded_lbl = [-100] * padding_length + lbl
            else:
                # Right padding: append -100 tokens
                padded_lbl = lbl + [-100] * padding_length
            padded_labels.append(padded_lbl)
        batch_labels = torch.tensor(padded_labels, dtype=torch.long)

        return {
            "input_ids": batch_input["input_ids"],
            "attention_mask": batch_input["attention_mask"],
            "labels": batch_labels,
        }


def main():
    parser = argparse.ArgumentParser()

    # Paths (single dataset mode - original)
    parser.add_argument("--database_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--table_value_cache_path", type=str, default=None)
    parser.add_argument("--table_info_cache_path", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="ra_sft_chat_model")

    # Multi-dataset mode: combine Spider and BIRD
    parser.add_argument("--combine_datasets", action="store_true",
                       help="Combine multiple datasets (e.g., Spider + BIRD)")
    # Spider paths
    parser.add_argument("--spider_database_path", type=str, default=None)
    parser.add_argument("--spider_table_value_cache_path", type=str, default=None)
    parser.add_argument("--spider_table_info_cache_path", type=str, default=None)
    parser.add_argument("--spider_input_file", type=str, default=None)
    # BIRD paths
    parser.add_argument("--bird_database_path", type=str, default=None)
    parser.add_argument("--bird_table_value_cache_path", type=str, default=None)
    parser.add_argument("--bird_table_info_cache_path", type=str, default=None)
    parser.add_argument("--bird_input_file", type=str, default=None)

    # Model & tokenizer
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--task_type", type=str, choices=["ra", "sql", "ra_sql", "cot"], default="ra")
    parser.add_argument("--system_message", type=str, default=DEFAULT_SYSTEM_MESSAGE)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--serialize_cot_type", type=str, default="")
    parser.add_argument("--full_loss", action="store_true")
    parser.add_argument("--train_split", type=float, default=0.95)
    parser.add_argument("--debug_dataset", action="store_true")
    parser.add_argument("--sel_seed", type=int, default=None)
    parser.add_argument("--num_prompt_examples", type=int, default=5, help="Number of prompt examples to log during training")

    # Training hyperparams
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_limit", type=int, default=2)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--left_padding", action="store_true", help="Use left padding instead of right padding (useful for some models)")
    parser.add_argument("--train_mode", type=str, default="train")
    parser.add_argument("--apply_template", action="store_true", help="Whether to apply chat template (should be true for chat models)")
    parser.add_argument("--cached_dataset_path", type=str, default=None,
                       help="Path to pre-built cached dataset (skips slow dataset building)")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training")
    args = parser.parse_args()

    import os

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
            project="relational-algebra-sft-chat",
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

    # Load tokenizer with chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side based on argument
    if args.left_padding:
        tokenizer.padding_side = "left"
        print("Using left padding for training")
    else:
        tokenizer.padding_side = "right"
        print("Using right padding for training")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Load dataset(s)
    if args.combine_datasets:
        # Multi-dataset mode
        print("=== Multi-dataset mode: combining datasets ===")
        combined_datasets_config = []

        # Load Spider dataset if specified
        if args.spider_input_file:
            spider_data = json.load(open(args.spider_input_file))
            split_idx = int(len(spider_data) * args.train_split)
            spider_train = spider_data[:split_idx]
            spider_val = spider_data[split_idx:]

            combined_datasets_config.append({
                'input_dataset': spider_train,
                'database_path': args.spider_database_path,
                'dataset_name': 'spider',
                'table_value_cache_path': args.spider_table_value_cache_path,
                'table_info_cache_path': args.spider_table_info_cache_path,
            })
            print(f"  - Spider: {len(spider_train)} train samples")
        else:
            spider_val = []

        # Load BIRD dataset if specified
        if args.bird_input_file:
            bird_data = json.load(open(args.bird_input_file))
            split_idx = int(len(bird_data) * args.train_split)
            bird_train = bird_data[:split_idx]
            bird_val = bird_data[split_idx:]

            combined_datasets_config.append({
                'input_dataset': bird_train,
                'database_path': args.bird_database_path,
                'dataset_name': 'bird',
                'table_value_cache_path': args.bird_table_value_cache_path,
                'table_info_cache_path': args.bird_table_info_cache_path,
            })
            print(f"  - BIRD: {len(bird_train)} train samples")
        else:
            bird_val = []

        total_train = sum(len(cfg['input_dataset']) for cfg in combined_datasets_config)
        print(f"  - Total training samples: {total_train}")

        # Create combined training dataset
        train_dataset = ChatAlignedRelationalQueryDataset(
            input_dataset=None,  # Not used when combined_datasets is provided
            database_path=None,
            dataset_name=None,
            table_value_cache_path=None,
            table_info_cache_path=None,
            tokenizer=tokenizer,
            max_length=args.max_length,
            mode=args.train_mode,
            debug_max_items=32 if args.debug_dataset else None,
            cot=args.cot,
            serialize_cot_type=args.serialize_cot_type,
            full_loss=args.full_loss,
            task_type=args.task_type,
            system_message=args.system_message,
            sel_seed=args.sel_seed,
            save_prompt_examples=args.num_prompt_examples,
            apply_template=args.apply_template,
            combined_datasets=combined_datasets_config,
        )

        # Create combined validation dataset (combine Spider and BIRD val sets)
        if spider_val or bird_val:
            combined_val_config = []
            if spider_val:
                combined_val_config.append({
                    'input_dataset': spider_val,
                    'database_path': args.spider_database_path,
                    'dataset_name': 'spider',
                    'table_value_cache_path': args.spider_table_value_cache_path,
                    'table_info_cache_path': args.spider_table_info_cache_path,
                })
            if bird_val:
                combined_val_config.append({
                    'input_dataset': bird_val,
                    'database_path': args.bird_database_path,
                    'dataset_name': 'bird',
                    'table_value_cache_path': args.bird_table_value_cache_path,
                    'table_info_cache_path': args.bird_table_info_cache_path,
                })

            val_dataset = ChatAlignedRelationalQueryDataset(
                input_dataset=None,
                database_path=None,
                dataset_name=None,
                table_value_cache_path=None,
                table_info_cache_path=None,
                tokenizer=tokenizer,
                max_length=args.max_length,
                mode='dev',
                debug_max_items=16 if args.debug_dataset else None,
                cot=args.cot,
                serialize_cot_type=args.serialize_cot_type,
                full_loss=args.full_loss,
                task_type=args.task_type,
                system_message=args.system_message,
                sel_seed=args.sel_seed,
                apply_template=args.apply_template,
                combined_datasets=combined_val_config,
            )
        else:
            val_dataset = None

    else:
        # Single dataset mode (original behavior)
        print("=== Single dataset mode ===")

        # Check if we should load from pre-built cache
        if args.cached_dataset_path:
            import pickle
            print(f"Loading pre-built dataset from {args.cached_dataset_path}...")
            with open(args.cached_dataset_path, 'rb') as f:
                full_dataset = pickle.load(f)

            # Split into train/val
            split_idx = int(len(full_dataset.samples) * args.train_split)
            train_samples = full_dataset.samples[:split_idx]
            val_samples = full_dataset.samples[split_idx:]

            # Create train dataset with cached samples
            train_dataset = ChatAlignedRelationalQueryDataset.__new__(ChatAlignedRelationalQueryDataset)
            train_dataset.samples = train_samples
            train_dataset.tokenizer = tokenizer
            train_dataset.max_length = args.max_length
            train_dataset.task_type = args.task_type
            train_dataset.full_loss = args.full_loss
            train_dataset.system_message = args.system_message
            train_dataset.sel_seed = args.sel_seed
            train_dataset.save_prompt_examples = args.num_prompt_examples
            train_dataset.prompt_examples = []

            # Create val dataset with cached samples
            val_dataset = ChatAlignedRelationalQueryDataset.__new__(ChatAlignedRelationalQueryDataset)
            val_dataset.samples = val_samples
            val_dataset.tokenizer = tokenizer
            val_dataset.max_length = args.max_length
            val_dataset.task_type = args.task_type
            val_dataset.full_loss = args.full_loss
            val_dataset.system_message = args.system_message
            val_dataset.sel_seed = args.sel_seed
            val_dataset.save_prompt_examples = 0
            val_dataset.prompt_examples = []

            print(f"✓ Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples from cache")

        else:
            # Original dataset building code
            if not all([args.input_file, args.database_path, args.dataset_name,
                        args.table_value_cache_path, args.table_info_cache_path]):
                raise ValueError("In single dataset mode, must provide: --input_file, --database_path, "
                               "--dataset_name, --table_value_cache_path, --table_info_cache_path")

            input_dataset = json.load(open(args.input_file))
            split_idx = int(len(input_dataset) * args.train_split)
            train_data = input_dataset[:split_idx]
            val_data = input_dataset[split_idx:]

            train_dataset = ChatAlignedRelationalQueryDataset(
                input_dataset=train_data,
                database_path=args.database_path,
                dataset_name=args.dataset_name,
                table_value_cache_path=args.table_value_cache_path,
                table_info_cache_path=args.table_info_cache_path,
                tokenizer=tokenizer,
                max_length=args.max_length,
                mode=args.train_mode,
                debug_max_items=32 if args.debug_dataset else None,
                cot=args.cot,
                serialize_cot_type=args.serialize_cot_type,
                full_loss=args.full_loss,
                task_type=args.task_type,
                system_message=args.system_message,
                sel_seed=args.sel_seed,
                save_prompt_examples=args.num_prompt_examples,
                apply_template=args.apply_template,
            )

            val_dataset = ChatAlignedRelationalQueryDataset(
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
                system_message=args.system_message,
                sel_seed=args.sel_seed,
                apply_template=args.apply_template,
            )

    # Save prompt examples to file
    if args.num_prompt_examples > 0 and len(train_dataset.prompt_examples) > 0:
        prompt_examples_path = os.path.join(args.output_dir, "prompt_examples.json")
        with open(prompt_examples_path, "w") as f:
            json.dump(train_dataset.prompt_examples, f, indent=2)
        print(f"Saved {len(train_dataset.prompt_examples)} prompt examples to: {prompt_examples_path}")

    collator = DataCollatorForCausalLMMasking(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Save padded prompt examples (showing actual batched/padded sequences)
    if args.num_prompt_examples > 0 and len(train_dataset) > 0:
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
                "num_padding_tokens": attention_mask.count(0) if tokenizer.padding_side == "right" else attention_mask.count(0),
                "padding_side": tokenizer.padding_side,
            })

        padded_examples_path = os.path.join(args.output_dir, "padded_prompt_examples.json")
        with open(padded_examples_path, "w") as f:
            json.dump(padded_examples, f, indent=2)
        print(f"Saved {len(padded_examples)} padded prompt examples to: {padded_examples_path}")

    trainer.train()
    trainer.save_state()
    trainer.save_model(args.output_dir)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

