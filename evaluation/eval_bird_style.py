#!/usr/bin/env python3
"""
Official BIRD-style execution evaluation

This module implements the official BIRD benchmark execution evaluation
using simple set comparison (as opposed to Spider's res_map comparison).

Key difference from Spider evaluation (eval_spider.py):
- BIRD: Uses set(results) comparison - ignores row order and duplicates
- Spider: Uses res_map comparison - more strict, catches duplicates

This gives BIRD evaluation generally 2-5% higher execution accuracy.
"""

import sqlite3
from typing import Tuple, Optional
import multiprocessing
import traceback


def _execute_sql_worker(db_path, sql_query, result_queue, error_queue):
    """
    Worker function that executes SQL in a separate process.

    This allows us to forcefully terminate long-running queries,
    which is impossible with threading due to SQLite's blocking C operations.

    Args:
        db_path: Path to SQLite database
        sql_query: SQL query to execute
        result_queue: Queue to put results in
        error_queue: Queue to put errors in
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        result_queue.put(results)
    except Exception as e:
        error_queue.put({
            'type': type(e).__name__,
            'message': str(e),
            'traceback': traceback.format_exc()
        })


def _execute_sql_with_timeout(db_path: str, sql_query: str, timeout: Optional[int]) -> Tuple[Optional[list], Optional[str]]:
    """
    Execute SQL with process-based timeout.

    When timeout=None: Executes directly (IDENTICAL to original behavior)
    When timeout is set: Uses multiprocessing.Process which can be forcefully terminated

    Args:
        db_path: Path to SQLite database
        sql_query: SQL query to execute
        timeout: Timeout in seconds (None = no timeout)

    Returns:
        Tuple of (results, error_message)
        - (results, None) on success
        - (None, error_msg) on failure
    """
    if timeout is None:
        # No timeout - execute directly (100% IDENTICAL to original behavior)
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            conn.close()
            return results, None
        except sqlite3.Error as e:
            return None, f"SQL execution error: {str(e)}"
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {str(e)}"

    # With timeout - use multiprocessing to allow termination
    result_queue = multiprocessing.Queue()
    error_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=_execute_sql_worker,
        args=(db_path, sql_query, result_queue, error_queue)
    )

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        # Timeout! Process is still running - kill it
        process.terminate()
        process.join(timeout=1)  # Give it 1 second to terminate gracefully

        if process.is_alive():
            # Still alive - force kill
            process.kill()
            process.join()

        return None, f"SQL execution timeout after {timeout}s"

    # Process finished - check for results or errors
    if not error_queue.empty():
        error = error_queue.get()
        return None, f"{error['type']}: {error['message']}"

    if not result_queue.empty():
        results = result_queue.get()
        return results, None

    # Process finished but no results/errors (shouldn't happen)
    return None, "Unknown error: process finished without results"


def eval_exec_match_bird_style(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
    timeout: int = None
) -> Tuple[bool, Optional[str]]:
    """
    Official BIRD-style execution evaluation using simple set comparison.

    *** EVALUATION LOGIC IS 100% UNCHANGED - ONLY TIMEOUT MECHANISM ADDED ***

    Args:
        db_path: Path to SQLite database file
        pred_sql: Predicted SQL query string
        gold_sql: Ground truth SQL query string
        timeout: Timeout in seconds (default: None = no timeout, original behavior)

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
        - (True, None) if execution matches
        - (False, error_msg) if execution fails or doesn't match

    Examples:
        >>> eval_exec_match_bird_style("db.sqlite", "SELECT * FROM users", "SELECT * FROM users")
        (True, None)

        >>> # Row order differences are IGNORED (set comparison)
        >>> eval_exec_match_bird_style("db.sqlite",
        ...     "SELECT name FROM users ORDER BY id",
        ...     "SELECT name FROM users ORDER BY id DESC")
        (True, None)  # If same names, just different order

        >>> # Duplicate rows are IGNORED (set comparison)
        >>> eval_exec_match_bird_style("db.sqlite",
        ...     "SELECT name FROM users",  # returns ['Alice', 'Bob', 'Alice']
        ...     "SELECT DISTINCT name FROM users")  # returns ['Alice', 'Bob']
        (True, None)  # set() removes duplicates
    """
    try:
        # Execute predicted SQL (with process-based timeout if specified)
        pred_res, pred_error = _execute_sql_with_timeout(db_path, pred_sql, timeout)
        if pred_error:
            return False, f"Predicted SQL: {pred_error}"

        # Execute ground truth SQL (with process-based timeout if specified)
        gold_res, gold_error = _execute_sql_with_timeout(db_path, gold_sql, timeout)
        if gold_error:
            return False, f"Gold SQL: {gold_error}"

        # ============================================================================
        # OFFICIAL BIRD EVALUATION LOGIC - 100% UNCHANGED
        # ============================================================================
        # BIRD-style comparison: Convert to sets and compare
        # This ignores:
        # - Row order: set([('Alice', 25), ('Bob', 30)]) == set([('Bob', 30), ('Alice', 25)])
        # - Duplicates: set([('Alice',), ('Bob',), ('Alice',)]) == set([('Alice',), ('Bob',)])
        try:
            pred_set = set(pred_res)
            gold_set = set(gold_res)

            if pred_set == gold_set:
                return True, None
            else:
                # Provide helpful error message
                pred_only = pred_set - gold_set
                gold_only = gold_set - pred_set
                error_parts = []
                if pred_only:
                    error_parts.append(f"Predicted has extra rows: {list(pred_only)[:3]}")
                if gold_only:
                    error_parts.append(f"Gold has extra rows: {list(gold_only)[:3]}")
                return False, "; ".join(error_parts)
        except TypeError as e:
            # Unhashable type in results (e.g., lists, dicts)
            return False, f"Results not comparable (unhashable type): {str(e)}"
        # ============================================================================
        # END OF OFFICIAL BIRD EVALUATION LOGIC
        # ============================================================================

    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {str(e)}"


def eval_exec_match_bird_style_simple(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
    timeout: int = None
) -> bool:
    """
    Simplified version that returns only boolean result.

    Args:
        db_path: Path to SQLite database file
        pred_sql: Predicted SQL query string
        gold_sql: Ground truth SQL query string
        timeout: Timeout in seconds (default: None = no timeout)

    Returns:
        True if queries match, False otherwise
    """
    success, _ = eval_exec_match_bird_style(db_path, pred_sql, gold_sql, timeout)
    return success


def eval_exec_match_bird_style_with_results(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
    timeout: int = None
):
    """
    BIRD-style execution evaluation that returns actual query results.

    This is used when you need to inspect the actual results, not just
    whether they match.

    Args:
        db_path: Path to SQLite database file
        pred_sql: Predicted SQL query string
        gold_sql: Ground truth SQL query string
        timeout: Timeout in seconds (default: None = no timeout)

    Returns:
        Tuple of (success, error, pred_results, gold_results)
        - success: bool - True if results match
        - error: Optional[str] - Error message if failed
        - pred_results: list - Predicted query results (or None if failed)
        - gold_results: list - Gold query results (or None if failed)
    """
    try:
        # Execute predicted SQL with process-based timeout
        pred_res, pred_error = _execute_sql_with_timeout(db_path, pred_sql, timeout)
        if pred_error:
            return False, f"Predicted SQL: {pred_error}", None, None

        # Execute ground truth SQL with process-based timeout
        gold_res, gold_error = _execute_sql_with_timeout(db_path, gold_sql, timeout)
        if gold_error:
            return False, f"Gold SQL: {gold_error}", pred_res, None

        # ============================================================================
        # OFFICIAL BIRD EVALUATION LOGIC - 100% UNCHANGED
        # ============================================================================
        # BIRD-style comparison: Convert to sets and compare
        try:
            pred_set = set(pred_res)
            gold_set = set(gold_res)

            if pred_set == gold_set:
                return True, None, pred_res, gold_res
            else:
                # Provide helpful error message
                pred_only = pred_set - gold_set
                gold_only = gold_set - pred_set
                error_parts = []
                if pred_only:
                    error_parts.append(f"Predicted has extra rows: {list(pred_only)[:3]}")
                if gold_only:
                    error_parts.append(f"Gold has extra rows: {list(gold_only)[:3]}")
                return False, "; ".join(error_parts), pred_res, gold_res
        except TypeError as e:
            # Unhashable type in results (e.g., lists, dicts)
            return False, f"Results not comparable (unhashable type): {str(e)}", pred_res, gold_res
        # ============================================================================
        # END OF OFFICIAL BIRD EVALUATION LOGIC
        # ============================================================================

    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {str(e)}", None, None
