"""
Spider2 evaluation module.

Spider2 uses a different evaluation approach:
- No gold SQL is provided
- Gold execution results (CSV files) are pre-computed
- Predicted SQL is executed and results are compared against gold CSVs
- Comparison uses tolerances and can ignore column order

Based on OmniSQL's evaluate_spider2.py
"""

import json
import os
import re
import math
import sqlite3
import tempfile
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Spider2 evaluation will not work.")

try:
    from func_timeout import func_timeout, FunctionTimedOut
    FUNC_TIMEOUT_AVAILABLE = True
except ImportError:
    FUNC_TIMEOUT_AVAILABLE = False
    print("Warning: func_timeout not available. Using no timeout for Spider2 evaluation.")


def load_jsonl_to_dict(jsonl_file: str) -> Dict[str, Any]:
    """Load JSONL file into a dictionary keyed by instance_id."""
    data_dict = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data_dict[item['instance_id']] = item
    return data_dict


def compare_pandas_table(pred: 'pd.DataFrame', gold: 'pd.DataFrame',
                         condition_cols: List[int] = None,
                         ignore_order: bool = False,
                         tolerance: float = 1e-2) -> int:
    """
    Compare predicted DataFrame with gold DataFrame.

    Args:
        pred: Predicted execution result as DataFrame
        gold: Gold execution result as DataFrame
        condition_cols: Column indices to compare (None = all columns)
        ignore_order: Whether to ignore row order
        tolerance: Tolerance for numeric comparisons

    Returns:
        1 if match, 0 if not
    """
    if not PANDAS_AVAILABLE:
        return 0

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        if ignore_order_:
            v1 = sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
            v2 = sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    if condition_cols is not None and condition_cols != []:
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()

    score = 1
    for gold_col in t_gold_list:
        if not any(vectors_match(gold_col, pred_col, ignore_order_=ignore_order)
                   for pred_col in t_pred_list):
            score = 0
            break

    return score


def compare_multi_pandas_table(pred: 'pd.DataFrame',
                               multi_gold: List['pd.DataFrame'],
                               multi_condition_cols: List[List[int]],
                               multi_ignore_order: List[bool]) -> int:
    """
    Compare predicted DataFrame against multiple possible gold DataFrames.
    Returns 1 if any gold matches.
    """
    if multi_condition_cols == [] or multi_condition_cols == [[]] or \
       multi_condition_cols == [None] or multi_condition_cols is None:
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(sublist, list) for sublist in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]

    assert len(multi_gold) == len(multi_condition_cols) == len(multi_ignore_order)

    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]):
            return 1
    return 0


def get_sqlite_result(db_file_path: str, query: str,
                      timeout: int = 30) -> Tuple[bool, Any]:
    """
    Execute SQL query on SQLite database and return results as DataFrame.

    Args:
        db_file_path: Path to SQLite database
        query: SQL query to execute
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, result_df or error_message)
    """
    if not PANDAS_AVAILABLE:
        return False, "pandas not available"

    def _execute():
        conn = sqlite3.connect(db_file_path)
        memory_conn = sqlite3.connect(':memory:')
        conn.backup(memory_conn)
        try:
            df = pd.read_sql_query(query, memory_conn)
            return True, df
        except Exception as e:
            return False, str(e)
        finally:
            memory_conn.close()
            conn.close()

    if FUNC_TIMEOUT_AVAILABLE:
        try:
            return func_timeout(timeout, _execute)
        except FunctionTimedOut:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
    else:
        try:
            return _execute()
        except Exception as e:
            return False, str(e)


def evaluate_single_spider2(
    instance_id: str,
    pred_sql: str,
    db_path: str,
    db_id: str,
    gold_result_dir: str,
    eval_standard: Dict[str, Any],
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Evaluate a single Spider2 prediction.

    Args:
        instance_id: Instance identifier
        pred_sql: Predicted SQL query
        db_path: Path to database directory
        db_id: Database ID
        gold_result_dir: Directory containing gold execution results
        eval_standard: Evaluation standard (condition_cols, ignore_order)
        timeout: Execution timeout

    Returns:
        Dictionary with instance_id, score, pred_sql, error_info
    """
    error_info = None

    # Execute predicted SQL
    db_file_path = os.path.join(db_path, db_id, f"{db_id}.sqlite")
    success, result = get_sqlite_result(db_file_path, pred_sql, timeout)

    if not success:
        return {
            "instance_id": instance_id,
            "score": 0,
            "pred_sql": pred_sql,
            "error_info": f"Execution error: {result}"
        }

    pred_df = result

    # Find gold result files
    pattern = re.compile(rf'^{re.escape(instance_id)}(_[a-z])?\.csv$')
    all_files = os.listdir(gold_result_dir)
    csv_files = [f for f in all_files if pattern.match(f)]

    if len(csv_files) == 0:
        return {
            "instance_id": instance_id,
            "score": 0,
            "pred_sql": pred_sql,
            "error_info": f"No gold result file found for {instance_id}"
        }

    try:
        condition_cols = eval_standard.get('condition_cols', [])
        ignore_order = eval_standard.get('ignore_order', False)

        if len(csv_files) == 1:
            gold_df = pd.read_csv(os.path.join(gold_result_dir, f"{instance_id}.csv"))
            score = compare_pandas_table(pred_df, gold_df, condition_cols, ignore_order)
        else:
            # Multiple possible gold results
            gold_dfs = [pd.read_csv(os.path.join(gold_result_dir, f)) for f in csv_files]
            ignore_orders = [ignore_order] * len(gold_dfs)
            score = compare_multi_pandas_table(pred_df, gold_dfs,
                                               [condition_cols] * len(gold_dfs),
                                               ignore_orders)

        if score == 0:
            error_info = "Result mismatch"

    except Exception as e:
        score = 0
        error_info = f"Comparison error: {str(e)}"

    return {
        "instance_id": instance_id,
        "score": score,
        "pred_sql": pred_sql,
        "error_info": error_info
    }


def evaluate_spider2_batch(
    predictions: List[Tuple[str, str, str]],  # List of (instance_id, pred_sql, db_id)
    db_path: str,
    gold_result_dir: str,
    eval_standard_path: str,
    timeout: int = 30,
    verbose: bool = False
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate a batch of Spider2 predictions.

    Args:
        predictions: List of (instance_id, pred_sql, db_id) tuples
        db_path: Path to database directory
        gold_result_dir: Directory containing gold execution results
        eval_standard_path: Path to eval_standard.jsonl
        timeout: Execution timeout per query
        verbose: Whether to print progress

    Returns:
        Tuple of (accuracy, detailed_results)
    """
    if not PANDAS_AVAILABLE:
        print("Error: pandas is required for Spider2 evaluation")
        return 0.0, []

    # Load eval standard
    eval_standard_dict = load_jsonl_to_dict(eval_standard_path)

    results = []
    for instance_id, pred_sql, db_id in tqdm(predictions, desc="Evaluating Spider2", disable=not verbose):
        eval_standard = eval_standard_dict.get(instance_id, {
            'condition_cols': [],
            'ignore_order': False
        })

        result = evaluate_single_spider2(
            instance_id=instance_id,
            pred_sql=pred_sql,
            db_path=db_path,
            db_id=db_id,
            gold_result_dir=gold_result_dir,
            eval_standard=eval_standard,
            timeout=timeout
        )
        results.append(result)

        if verbose:
            status = "✓" if result['score'] == 1 else "✗"
            print(f"  {status} {instance_id}: score={result['score']}")

    # Calculate accuracy
    total_score = sum(r['score'] for r in results)
    accuracy = total_score / len(results) if results else 0.0

    return accuracy, results


def run_spider2_eval(
    gold_file: str,
    pred_file: str,
    db_path: str,
    gold_result_dir: str,
    eval_standard_path: str,
    timeout: int = 30,
    verbose: bool = False
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run Spider2 evaluation in the same format as OmniSQL.

    Args:
        gold_file: Path to gold file (test.json with instance_id, db_id, question)
        pred_file: Path to predictions file (JSON with pred_sqls)
        db_path: Path to database directory
        gold_result_dir: Directory containing gold execution results
        eval_standard_path: Path to eval_standard.jsonl
        timeout: Execution timeout per query
        verbose: Whether to print progress

    Returns:
        Tuple of (accuracy, detailed_results)
    """
    # Load gold data
    with open(gold_file, 'r') as f:
        gold_data = json.load(f)

    # Load predictions
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    assert len(gold_data) == len(pred_data), \
        f"Length mismatch: {len(gold_data)} vs {len(pred_data)}"

    # Prepare predictions list
    predictions = []
    pred_sql_key = "pred_sqls" if "pred_sqls" in pred_data[0] else "pred_sql"

    for gold, pred in zip(gold_data, pred_data):
        instance_id = gold['instance_id']
        db_id = gold['db_id']

        # Get first prediction (greedy decoding)
        if isinstance(pred.get(pred_sql_key), list):
            pred_sql = pred[pred_sql_key][0]
        else:
            pred_sql = pred.get(pred_sql_key, pred.get('pred_sql', ''))

        predictions.append((instance_id, pred_sql, db_id))

    return evaluate_spider2_batch(
        predictions=predictions,
        db_path=db_path,
        gold_result_dir=gold_result_dir,
        eval_standard_path=eval_standard_path,
        timeout=timeout,
        verbose=verbose
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spider2 Evaluation")
    parser.add_argument("--gold", required=True, help="Path to gold file")
    parser.add_argument("--pred", required=True, help="Path to predictions file")
    parser.add_argument("--db_path", required=True, help="Path to database directory")
    parser.add_argument("--gold_result_dir", required=True, help="Path to gold execution results")
    parser.add_argument("--eval_standard", required=True, help="Path to eval_standard.jsonl")
    parser.add_argument("--timeout", type=int, default=30, help="Query timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    accuracy, results = run_spider2_eval(
        gold_file=args.gold,
        pred_file=args.pred,
        db_path=args.db_path,
        gold_result_dir=args.gold_result_dir,
        eval_standard_path=args.eval_standard,
        timeout=args.timeout,
        verbose=args.verbose
    )

    print(f"\nSpider2 Execution Accuracy: {accuracy:.4f}")
    print(f"Correct: {sum(r['score'] for r in results)}/{len(results)}")
