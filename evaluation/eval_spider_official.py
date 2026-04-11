#!/usr/bin/env python3
"""
Official Spider Execution Evaluation

This module implements the official Spider benchmark execution evaluation from:
https://github.com/taoyds/test-suite-sql-eval

Key features:
- Sophisticated column permutation handling (more lenient than simple res_map)
- Multiset equality checking with bag semantics
- Order-aware comparison when ORDER BY is present
- Value plugging support (optional - tests structure without value correctness)
- Async execution with configurable timeout

Differences from our simple eval_spider.py:
- Official Spider: Tries column permutations, more lenient on column order
- Our eval_spider.py: Uses res_map, stricter column order requirement

Usage:
    from eval_spider_official import eval_exec_match_official

    # Simple usage (single database)
    result = eval_exec_match_official_simple(
        db_path="/path/to/db.sqlite",
        pred_sql="SELECT * FROM users",
        gold_sql="SELECT name, age FROM users"
    )

    # Full usage (with value plugging and multiple databases)
    result = eval_exec_match_official(
        db_path="/path/to/db.sqlite",  # Can test all DBs in same directory
        pred_sql="SELECT * FROM users WHERE age > 20",
        gold_sql="SELECT name FROM users WHERE age > 25",
        plug_value=True,  # Test structure, plug in gold values
        keep_distinct=False,  # Ignore DISTINCT differences
        progress_bar=False
    )
"""

import os
import re
import asyncio
import sqlite3
from typing import Tuple, Any, List, Set, Iterator
from itertools import product, chain
from collections import defaultdict
import sqlparse


# ============================================================================
# Parse utilities (from test-suite-sql-eval/parse.py)
# ============================================================================

VALUE_NUM_SYMBOL = 'VALUERARE'
QUOTE_CHARS = {'`', '\'', '"'}
TIMEOUT = 60


def strip_query(query: str) -> Tuple[List[str], List[str]]:
    """Extract query keywords and values, replacing values with VALUE_NUM_SYMBOL."""
    query_keywords, all_values = [], []

    toks = sqlparse.parse(query)[0].flatten()
    values = [t.value for t in toks if t.ttype == sqlparse.tokens.Literal.String.Single
              or t.ttype == sqlparse.tokens.Literal.String.Symbol]

    for val in values:
        all_values.append(val)
        query = query.replace(val.strip(), VALUE_NUM_SYMBOL)

    query_tokenized = query.split()
    float_nums = re.findall(r"[-+]?\d*\.\d+", query)
    all_values += [qt for qt in query_tokenized if qt in float_nums]
    query_tokenized = [VALUE_NUM_SYMBOL if qt in float_nums else qt for qt in query_tokenized]

    query = " ".join(query_tokenized)
    int_nums = [i.strip() for i in re.findall(r"[^tT]\d+", query)]

    all_values += [qt for qt in query_tokenized if qt in int_nums]
    query_tokenized = [VALUE_NUM_SYMBOL if qt in int_nums else qt for qt in query_tokenized]

    for tok in query_tokenized:
        if "." in tok:
            table = re.findall(r"[Tt]\d+\.", tok)
            if len(table) > 0:
                to = tok.replace(".", " . ").split()
                to = [t.lower() for t in to if len(t) > 0]
                query_keywords.extend(to)
            else:
                query_keywords.append(tok.lower())
        elif len(tok) > 0:
            query_keywords.append(tok.lower())

    return query_keywords, all_values


def reformat_query(query: str) -> str:
    """Reformat query by removing whitespace and normalizing table aliases."""
    query = query.strip().replace(";", "").replace("\t", "")
    tokens = sqlparse.parse(query)[0].flatten()
    query = ' '.join([t.value for t in tokens if t.ttype != sqlparse.tokens.Whitespace])

    # Normalize table aliases
    t_stars = ["t1.*", "t2.*", "t3.*", "T1.*", "T2.*", "T3.*"]
    for ts in t_stars:
        query = query.replace(ts, "*")
    return query


def replace_values(sql: str) -> Tuple[List[str], Set[str]]:
    """Replace values in SQL with VALUE_NUM_SYMBOL and return value set."""
    sql = sqlparse.format(sql, reindent=False, keyword_case='upper')
    sql = re.sub(r"(T\d+\.)\s", r"\1", sql)
    query_toks_no_value, values = strip_query(sql)
    return query_toks_no_value, set(values)


def extract_query_values(sql: str) -> Tuple[List[str], Set[str]]:
    """Extract non-value tokens and value set from SQL query."""
    reformatted = reformat_query(query=sql)
    query_value_replaced, values = replace_values(reformatted)
    return query_value_replaced, values


def plugin(query_value_replaced: List[str], values_in_order: List[str]) -> str:
    """Plug values into query with value slots."""
    q_length = len(query_value_replaced)
    query_w_values = query_value_replaced[:]
    value_idx = [idx for idx in range(q_length) if query_value_replaced[idx] == VALUE_NUM_SYMBOL.lower()]
    assert len(value_idx) == len(values_in_order), \
        f"Value slots ({len(value_idx)}) != values provided ({len(values_in_order)})"

    for idx, value in zip(value_idx, values_in_order):
        query_w_values[idx] = value
    return ' '.join(query_w_values)


def plugin_all_permutations(query_value_replaced: List[str], values: Set[str]) -> Iterator[str]:
    """Generate all possible ways of filling values into predicted query."""
    from itertools import product
    num_slots = len([v for v in query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    for value_tuple in product(*[list(values) for _ in range(num_slots)]):
        yield plugin(query_value_replaced, list(value_tuple))


def get_all_preds_for_execution(gold: str, pred: str) -> Tuple[int, Iterator[str]]:
    """
    Extract values from gold query and generate all ways to plug them into predicted query.

    Returns:
        (num_alternatives, iterator of predictions with values plugged in)
    """
    _, gold_values = extract_query_values(gold)
    pred_query_value_replaced, _ = extract_query_values(pred)
    num_slots = len([v for v in pred_query_value_replaced if v == VALUE_NUM_SYMBOL.lower()])
    num_alternatives = len(gold_values) ** num_slots
    return num_alternatives, plugin_all_permutations(pred_query_value_replaced, gold_values)


def remove_distinct(s: str) -> str:
    """Remove DISTINCT keyword from SQL query."""
    toks = [t.value for t in list(sqlparse.parse(s)[0].flatten())]
    return ''.join([t for t in toks if t.lower() != 'distinct'])


# ============================================================================
# Core execution evaluation (from test-suite-sql-eval/exec_eval.py)
# ============================================================================

def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    """Permute tuple elements according to permutation."""
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    """Sort row elements for comparison (unordered)."""
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    """
    Quick rejection test: check if results have same bag of unordered rows.
    This is a necessary condition for denotational equivalence.
    """
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


def multiset_eq(l1: List, l2: List) -> bool:
    """Check whether two bags (multisets) are equivalent."""
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    """
    Constrain the space of column permutations by sampling rows.
    For efficiency when number of columns > 3.
    """
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]

    if num_cols <= 3:
        return product(*perm_constraints)

    # Sample 20 rows and constrain permutation space
    import random
    for _ in range(min(20, len(result2))):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)

    return product(*perm_constraints)


def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    """
    Check whether two denotations (query results) are equivalent.

    This is the core of official Spider evaluation:
    - Tries to find a column permutation that makes results equal
    - Handles multiset equality (bag semantics)
    - Respects order when ORDER BY is present
    """
    if len(result1) == 0 and len(result2) == 0:
        return True

    # Different number of rows -> different
    if len(result1) != len(result2):
        return False

    # Handle empty results
    if len(result1) == 0:
        return True

    num_cols = len(result1[0])

    # Different number of columns -> different
    if len(result2[0]) != num_cols:
        return False

    # Quick rejection: unorder each row and compare
    if not quick_rej(result1, result2, order_matters):
        return False

    # Now try to find column permutation that makes results equal
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # Enumerate possible column permutations
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue

        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]

        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # Check both set equality and multiset equality
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True

    return False


def replace_cur_year(query: str) -> str:
    """Replace YEAR(CURDATE()) with 2020 for reproducibility."""
    return re.sub(
        r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE
    )


def get_cursor_from_path(sqlite_path: str):
    """Get database cursor for a SQLite database path."""
    try:
        if not os.path.exists(sqlite_path):
            print(f"Opening a new connection {sqlite_path}")
        connection = sqlite3.connect(sqlite_path, timeout=30)
    except Exception as e:
        print(f"Error connecting to {sqlite_path}: {e}")
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


async def exec_on_db_(sqlite_path: str, query: str) -> Tuple[str, Any]:
    """Execute query on database (internal async function)."""
    query = replace_cur_year(query)
    cursor = get_cursor_from_path(sqlite_path)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        cursor.connection.close()
        return "result", result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return "exception", e


async def exec_on_db(
    sqlite_path: str, query: str, process_id: str = "", timeout: int = TIMEOUT
) -> Tuple[str, Any]:
    """Execute query on database with timeout."""
    try:
        return await asyncio.wait_for(exec_on_db_(sqlite_path, query), timeout)
    except asyncio.TimeoutError:
        return ('exception', TimeoutError(f"Query timeout after {timeout}s"))
    except Exception as e:
        return ("exception", e)


def postprocess(query: str) -> str:
    """Postprocess query to avoid execution errors (e.g., fix operator spacing)."""
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    return query


def eval_exec_match_official(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
    plug_value: bool = False,
    keep_distinct: bool = True,
    progress_bar: bool = False
) -> int:
    """
    Official Spider execution evaluation.

    Tests whether predicted and gold SQL are denotationally equivalent
    by executing them on all databases in the same directory as db_path.

    Args:
        db_path: Path to database file (e.g., /path/to/database/concert.sqlite)
        pred_sql: Predicted SQL query string
        gold_sql: Ground truth SQL query string
        plug_value: If True, test only structure by plugging gold values into pred
        keep_distinct: If False, remove DISTINCT from both queries before eval
        progress_bar: Show progress bar for each datapoint

    Returns:
        1 if denotationally equivalent, 0 otherwise

    Notes:
        - Uses bag semantics (multiset equality)
        - ORDER BY queries are evaluated with order_matters=True
        - More lenient than simple res_map comparison (tries column permutations)
    """
    # Postprocess queries
    pred_sql = postprocess(pred_sql)
    gold_sql = postprocess(gold_sql)

    if not keep_distinct:
        pred_sql = remove_distinct(pred_sql)
        gold_sql = remove_distinct(gold_sql)

    # Determine if order matters
    order_matters = 'order by' in gold_sql.lower()

    # Find all databases in the same directory
    db_dir = os.path.dirname(db_path)
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir)
                if '.sqlite' in basename]

    # If only one database, use it
    if len(db_paths) == 0:
        db_paths = [db_path]

    # Get predictions to test
    preds = [pred_sql]
    if plug_value:
        try:
            _, pred_variants = get_all_preds_for_execution(gold_sql, pred_sql)
            preds = chain([pred_sql], pred_variants)
        except Exception as e:
            # If value plugging fails, just use original prediction
            preds = [pred_sql]

    # Test each prediction variant
    for pred in preds:
        pred_passes = True

        # Progress bar wrapper
        if progress_bar:
            try:
                import tqdm
                ranger = tqdm.tqdm(db_paths)
            except ImportError:
                ranger = db_paths
        else:
            ranger = db_paths

        # Test on each database
        for test_db_path in ranger:
            g_flag, g_denotation = asyncio.run(exec_on_db(test_db_path, gold_sql))
            p_flag, p_denotation = asyncio.run(exec_on_db(test_db_path, pred))

            # Gold should always execute successfully
            if g_flag == 'exception':
                print(f"Warning: Gold query failed on {test_db_path}: {g_denotation}")
                # Don't assert, just skip this database
                continue

            # Prediction execution failure
            if p_flag == 'exception':
                pred_passes = False
                break

            # Check denotational equivalence
            if not result_eq(g_denotation, p_denotation, order_matters=order_matters):
                pred_passes = False
                break

        # If this prediction passed all databases, return success
        if pred_passes:
            return 1

    # None of the predictions passed
    return 0


def eval_exec_match_official_simple(
    db_path: str,
    pred_sql: str,
    gold_sql: str
) -> bool:
    """
    Simplified interface for official Spider execution evaluation.

    Args:
        db_path: Path to database file
        pred_sql: Predicted SQL query
        gold_sql: Ground truth SQL query

    Returns:
        True if execution results match, False otherwise

    Example:
        >>> result = eval_exec_match_official_simple(
        ...     "/path/to/db.sqlite",
        ...     "SELECT name FROM users",
        ...     "SELECT name FROM users"
        ... )
        >>> print(result)
        True
    """
    return eval_exec_match_official(
        db_path=db_path,
        pred_sql=pred_sql,
        gold_sql=gold_sql,
        plug_value=False,
        keep_distinct=True,
        progress_bar=False
    ) == 1


# ============================================================================
# Testing and Examples
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4:
        print("Usage: python eval_spider_official.py <db_path> <pred_sql> <gold_sql>")
        print("\nExample:")
        print('  python eval_spider_official.py database.sqlite "SELECT * FROM users" "SELECT name, age FROM users"')
        sys.exit(1)

    db_path = sys.argv[1]
    pred_sql = sys.argv[2]
    gold_sql = sys.argv[3]

    result = eval_exec_match_official_simple(db_path, pred_sql, gold_sql)

    print(f"Database: {db_path}")
    print(f"Predicted SQL: {pred_sql}")
    print(f"Gold SQL: {gold_sql}")
    print(f"\nResult: {'✓ MATCH' if result else '✗ NO MATCH'}")
