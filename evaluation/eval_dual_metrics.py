#!/usr/bin/env python3
"""
Dual Metrics Evaluation: Spider-style + BIRD-style

This module provides a unified interface to evaluate SQL execution using
both Spider-style (res_map) and BIRD-style (set comparison) methods.

Use this to see both metrics in your evaluation runs.
"""

from typing import Dict, Optional, Tuple
from eval_spider import evaluate_single_pair
from eval_bird_style import eval_exec_match_bird_style_simple
import os


def evaluate_sql_dual_metrics(
    pred_sql: str,
    gold_sql: str,
    db_dir: str,
    db_name: str,
    kmaps: dict,
    etype: str = 'all',
    dataset_name: str = 'spider',
    exec_timeout: int = None
) -> Dict:
    """
    Evaluate SQL using both Spider-style and BIRD-style execution metrics.

    Args:
        pred_sql: Predicted SQL query
        gold_sql: Ground truth SQL query
        db_dir: Base directory containing databases
        db_name: Database name/id
        kmaps: Foreign key maps
        etype: Evaluation type ('all', 'exec', or 'match')
        dataset_name: Dataset name ('spider', 'bird', etc.) for consistent exact match

    Returns:
        dict with structure:
        {
            'exact': int (0 or 1),
            'partial': dict,
            'hardness': str,
            'db_name': str,
            'exec_spider': int (0 or 1),  # Spider-style execution
            'exec_bird': int (0 or 1),    # BIRD-style execution
            'exec_methods_agree': bool,
            'exec_bird_more_lenient': bool
        }

    Notes:
        - For BIRD dataset: Uses string comparison for exact match (consistent across all samples)
        - For Spider dataset: Uses official Spider AST-based exact match
    """
    # For BIRD dataset, skip Spider parsing-based exact match for consistency
    # BIRD uses backtick syntax that Spider parser can't handle reliably
    # Using string comparison ensures all BIRD samples use the same exact match method
    if dataset_name.lower() == 'bird' and etype in ['all', 'match']:
        # Use string comparison for BIRD (consistent method for all samples)
        from eval_utils import normalize_sql_query

        gold_normalized = normalize_sql_query(gold_sql)
        pred_normalized = normalize_sql_query(pred_sql)
        exact_match = 1 if gold_normalized == pred_normalized else 0

        # Still get execution metrics but skip Spider parsing for exact match
        spider_result = evaluate_single_pair(
            pred_sql=pred_sql,
            gold_sql=gold_sql,
            db_dir=db_dir,
            db_name=db_name,
            etype='exec',  # Only execution, skip match to avoid parsing
            kmaps=kmaps,
            timeout=exec_timeout
        )
        # Override exact match with string comparison result
        spider_result['exact'] = exact_match
        spider_result['partial'] = None  # No partial scores for string comparison
    else:
        # Get Spider-style evaluation (includes exact match, partial, and Spider exec)
        spider_result = evaluate_single_pair(
            pred_sql=pred_sql,
            gold_sql=gold_sql,
            db_dir=db_dir,
            db_name=db_name,
            etype=etype,
            kmaps=kmaps,
            timeout=exec_timeout
        )

    result = {
        'exact': spider_result.get('exact'),
        'partial': spider_result.get('partial'),
        'hardness': spider_result.get('hardness'),
        'db_name': spider_result.get('db_name'),
        'exec_spider': spider_result.get('exec'),  # Rename for clarity
        'exec_bird': None,
        'exec_methods_agree': None,
        'exec_bird_more_lenient': None
    }

    # Add BIRD-style execution evaluation if exec was requested
    if etype in ['all', 'exec']:
        db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")

        try:
            bird_exec = eval_exec_match_bird_style_simple(db_path, pred_sql, gold_sql, timeout=exec_timeout)
            result['exec_bird'] = 1 if bird_exec else 0

            # Compare methods
            spider_exec = result['exec_spider']
            if spider_exec is not None:
                result['exec_methods_agree'] = (spider_exec == result['exec_bird'])
                result['exec_bird_more_lenient'] = (result['exec_bird'] == 1 and spider_exec == 0)
        except Exception as e:
            # If BIRD evaluation fails, keep it as None
            result['exec_bird'] = None
            result['exec_methods_agree'] = None
            result['exec_bird_more_lenient'] = None

    return result


def format_dual_metrics_summary(
    total: int,
    exec_spider_correct: int,
    exec_bird_correct: int,
    methods_agree: int,
    bird_more_lenient_cases: int
) -> str:
    """
    Format a summary string comparing Spider and BIRD execution metrics.

    Args:
        total: Total number of samples
        exec_spider_correct: Number correct with Spider-style evaluation
        exec_bird_correct: Number correct with BIRD-style evaluation
        methods_agree: Number of cases where both methods agree
        bird_more_lenient_cases: Number of cases where BIRD passes but Spider fails

    Returns:
        Formatted summary string
    """
    if total == 0:
        return "No samples evaluated"

    spider_acc = exec_spider_correct / total * 100
    bird_acc = exec_bird_correct / total * 100
    agreement_rate = methods_agree / total * 100
    lenient_rate = bird_more_lenient_cases / total * 100

    summary = []
    summary.append("\n" + "="*80)
    summary.append("EXECUTION ACCURACY COMPARISON: Spider-style vs BIRD-style")
    summary.append("="*80)
    summary.append(f"\n{'Metric':<40} {'Count':<10} {'Percentage':<10}")
    summary.append("-"*60)
    summary.append(f"{'Total Samples':<40} {total:<10} {100.0:<10.2f}%")
    summary.append(f"\n{'Spider-style Execution (current)':<40} {exec_spider_correct:<10} {spider_acc:<10.2f}%")
    summary.append(f"{'BIRD-style Execution (official)':<40} {exec_bird_correct:<10} {bird_acc:<10.2f}%")
    summary.append(f"{'Difference (BIRD - Spider)':<40} {exec_bird_correct - exec_spider_correct:<10} {bird_acc - spider_acc:<10.2f}%")
    summary.append(f"\n{'Both Methods Agree':<40} {methods_agree:<10} {agreement_rate:<10.2f}%")
    summary.append(f"{'BIRD More Lenient (passes when Spider fails)':<40} {bird_more_lenient_cases:<10} {lenient_rate:<10.2f}%")

    summary.append("\n" + "-"*80)
    summary.append("Interpretation:")
    summary.append("-"*80)

    if bird_acc > spider_acc:
        diff = bird_acc - spider_acc
        summary.append(f"✓ BIRD evaluation is MORE LENIENT: +{diff:.2f}% higher accuracy")
        summary.append("  → BIRD ignores row order and duplicate rows (uses set comparison)")
        summary.append("  → This is expected behavior for official BIRD benchmark")
        if diff > 5:
            summary.append(f"  ⚠️  Large difference ({diff:.2f}%) suggests many row order/duplicate issues")
    elif spider_acc > bird_acc:
        diff = spider_acc - bird_acc
        summary.append(f"⚠️  Spider evaluation is MORE LENIENT: +{diff:.2f}% higher accuracy")
        summary.append("  → This is UNUSUAL - Spider is typically more strict")
        summary.append("  → May indicate column ordering issues in predictions")
    else:
        summary.append("✓ Both methods agree perfectly")
        summary.append("  → No row order or duplicate issues in your predictions")

    summary.append("\n" + "="*80)

    return "\n".join(summary)


# Example usage
if __name__ == '__main__':
    print(__doc__)
    print("\nThis module is meant to be imported and used in evaluation scripts.")
    print("\nExample usage:")
    print("""
    from eval_dual_metrics import evaluate_sql_dual_metrics

    result = evaluate_sql_dual_metrics(
        pred_sql="SELECT * FROM users",
        gold_sql="SELECT * FROM users",
        db_dir="databases/",
        db_name="test_db",
        kmaps={},
        etype='all'
    )

    print(f"Spider execution: {result['exec_spider']}")
    print(f"BIRD execution: {result['exec_bird']}")
    print(f"Methods agree: {result['exec_methods_agree']}")
    """)
