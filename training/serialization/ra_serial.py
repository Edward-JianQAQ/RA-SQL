import json
import re
from copy import deepcopy

# --- 1. PRE-PROCESSING: Pointer Resolution for the New RA Format ---

POINTER_RE = re.compile(r"#\[(\d+)\.(\d+)\]")

def to_list(x):
    """A robust utility to convert a value to a list if it isn't already."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def resolve_ra_pointers(ra_tree: dict, schema_catalog: dict = None, annotate_schema: bool = False) -> dict:
    """
    Resolves abstract '#[d.i]' pointers in an RA tree into concrete column names.
    This is the crucial first step to make the RA tree interpretable.

    Args:
        ra_tree (dict): The raw RA plan tree with pointers.
        schema_catalog (dict): A mapping from table names to their column lists.
        annotate_schema (bool): If True, attaches the computed schema to each node.

    Returns:
        dict: A deep-copied RA tree with all pointers resolved.
    """
    tree = deepcopy(ra_tree)
    schema_cache = {}
    schema_catalog = schema_catalog or {}

    def get_descendant_by_depth(node, depth):
        current = node
        for _ in range(depth):
            if not current or not current.get("children"):
                return None
            current = current["children"][0]
        return current

    def substitute_pointers_in_value(value, current_node):
        if isinstance(value, str):
            def repl(m):
                depth, index = int(m.group(1)), int(m.group(2))
                target_node = get_descendant_by_depth(current_node, depth)
                if not target_node: return m.group(0)
                
                target_schema = compute_node_schema(target_node)
                if 0 <= index < len(target_schema):
                    return target_schema[index]
                return m.group(0)
            return POINTER_RE.sub(repl, value)
        
        if isinstance(value, list):
            return [substitute_pointers_in_value(v, current_node) for v in value]
        if isinstance(value, dict):
            return {k: substitute_pointers_in_value(v, current_node) for k, v in value.items()}
        return value

    def compute_node_schema(node):
        if not node: return []
        node_id = id(node)
        if node_id in schema_cache:
            return schema_cache[node_id]

        children = node.get("children", [])
        child_schemas = [compute_node_schema(c) for c in children]
        
        name = (node.get("name") or "").upper()
        einfo = node.get("extra_info", {})
        schema = []

        if name == "PROJECTION":
            resolved_exprs = substitute_pointers_in_value(einfo.get("Expressions", []), node)
            for expr in to_list(resolved_exprs):
                if isinstance(expr, dict) and 'expr' in expr:
                    schema.append(expr['expr'])
                else:
                    schema.append(str(expr))
        elif name in ("FILTER", "SORT", "DISTINCT"):
            schema = child_schemas[0] if child_schemas else []
        elif name == "AGGREGATE":
            resolved_exprs = substitute_pointers_in_value(einfo.get("Expressions", []), node)
            groups = substitute_pointers_in_value(einfo.get("Groups", []), node)
            # Handle groups - they might be dicts or strings
            group_cols = []
            for g in to_list(groups):
                if isinstance(g, dict) and 'expr' in g:
                    group_cols.append(g['expr'])
                else:
                    group_cols.append(str(g))
            # Handle expressions
            expr_cols = []
            for e in to_list(resolved_exprs):
                if isinstance(e, dict) and 'expr' in e:
                    expr_cols.append(e['expr'])
                else:
                    expr_cols.append(str(e))
            schema = group_cols + expr_cols
        elif "JOIN" in name:
            schema = [col for s in child_schemas for col in s]
        elif "SCAN" in name:
            table_name = einfo.get("Table")
            schema = schema_catalog.get(table_name, [])
        else: # UNION, INTERSECT, EXCEPT
            schema = child_schemas[0] if child_schemas else []

        schema_cache[node_id] = schema
        if annotate_schema:
            node["_schema"] = schema
        return schema

    def walk_and_resolve(node):
        for child in node.get("children", []):
            walk_and_resolve(child)
        
        if "extra_info" in node:
            node["extra_info"] = substitute_pointers_in_value(node["extra_info"], node)
        compute_node_schema(node)

    walk_and_resolve(tree)
    return tree

# --- 2. HELPER FUNCTIONS for Natural Language Generation ---

OPERATOR_NL_MAP = {
    'SEQ_SCAN': "retrieve all records",
    'FILTER': "filter these records",
    'PROJECTION': "select and keep only",
    'JOIN': "combine records from the two sources",
    'SORT': "sort the resulting records",
    'AGGREGATE': "group records and calculate",
    'LIMIT': "restrict the output",
    'DISTINCT': "remove any duplicate records",
    'UNION': "combine two sets of records",
    'INTERSECT': "find the common records between two sets",
    'EXCEPT': "take the first set of records and remove any from the second set",
}

def format_expression_list(expr_list):
    """Formats a list of columns or expressions into a readable string."""
    if not expr_list: return "the specified attributes"
    if len(expr_list) == 1: return f"the column '{expr_list[0]}'"
    return f"the columns '{', '.join(expr_list[:-1])}' and '{expr_list[-1]}'"

def parse_and_format_details(node):
    """
    (IMPROVED) Parses the 'extra_info' of a resolved node into a smooth NL phrase.
    Fixes bugs related to incorrect dictionary keys.
    """
    name = (node.get("name") or "").upper()
    info = node.get("extra_info", {})
    
    if "SCAN" in name:
        return f"from the table '{info.get('Table', 'unspecified')}'"
    if name == 'PROJECTION':
        exprs = to_list(info.get("Expressions", []))
        cols = [e['expr'] if isinstance(e, dict) else str(e) for e in exprs]
        return f"{format_expression_list(cols)}"
    if name == 'FILTER':
        # CORRECTED: Looks for 'Filters' or 'Conditions' and handles dict/str
        conditions = info.get('Filters', info.get('Conditions'))
        if isinstance(conditions, dict):
            return f"to keep only records where {conditions.get('expr', 'criteria are met')}"
        return f"to keep only records where {conditions or 'criteria are met'}"
    if name == 'JOIN':
        # CORRECTED: Looks for 'Condition' and handles dict/str
        condition = info.get('Condition', info.get('Conditions'))
        if isinstance(condition, dict):
            return f"based on the condition '{condition.get('expr', 'unspecified')}'"
        return f"based on the condition '{condition or 'unspecified'}'"
    if name == 'SORT':
        order_by = format_expression_list(to_list(info.get('Order', [])))
        limit = info.get('Limit')
        details = f"by {order_by}"
        if limit:
            details += f", keeping only the top {limit} record(s)"
        return details
    if name == 'AGGREGATE':
        # Handle groups - they might be dicts or strings
        groups_raw = to_list(info.get('Groups', []))
        groups = []
        for g in groups_raw:
            if isinstance(g, dict) and 'expr' in g:
                groups.append(g['expr'])
            elif isinstance(g, str):
                groups.append(g)

        # Handle expressions
        exprs_raw = to_list(info.get('Expressions', []))
        expr_strs = []
        for e in exprs_raw:
            if isinstance(e, dict) and 'expr' in e:
                expr_strs.append(e['expr'])
            else:
                expr_strs.append(str(e))

        aggs = format_expression_list(expr_strs)
        if groups:
            return f"{aggs}, grouped by {format_expression_list(groups)}"
        return f"{aggs}"
    if name == 'LIMIT':
        return f"to the top {info.get('Limit', 'N')} records"

    return "based on the specified parameters"

def generate_smooth_nl_rephrase(node):
    """Generates a complete narrative phrase for a single RA operation."""
    op_name = (node.get("name") or "").upper()
    op_phrase = OPERATOR_NL_MAP.get(op_name, f"perform the operation '{op_name}'")
    details = parse_and_format_details(node)
    return op_phrase, details

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation")))
from ra_eval_utils import (  # type: ignore
    split_conjuncts,
    parse_predicate,
    tokenize_ra_expression,
    _ensure_list_and_get_exprs,
)

# NEW: Render a parsed predicate (from ra_eval_utils.parse_predicate) into concise NL
def _render_predicate_to_text(p):
    kind = p.get("kind")
    if kind in {"is_null", "is_not_null"}:
        lhs = p.get("lhs_expr") or p.get("lhs") or "the field"
        return f"{lhs} is null" if kind == "is_null" else f"{lhs} is not null"
    if kind in {"in", "not_in"}:
        lhs = p.get("lhs") or "the field"
        rhs_list = p.get("rhs_list") or []
        rhs_text = ", ".join(str(x) for x in rhs_list)
        return f"{lhs} {'not in' if kind=='not_in' else 'in'} ({rhs_text})"
    if kind == "between":
        lhs = p.get("lhs") or "the field"
        lo, hi = p.get("range") or (None, None)
        return f"{lhs} between {lo} and {hi}"
    if kind == "like":
        lhs = p.get("lhs") or "the field"
        rhs = p.get("rhs")
        return f"{lhs} like {rhs}"
    if kind == "cmp":
        lhs = p.get("lhs") or "the field"
        op = p.get("op") or "="
        rhs = p.get("rhs")
        # Normalize operators for readability
        op_map = {"==": "=", "<>": "!=", None: "="}
        op = op_map.get(op, op)
        return f"{lhs} {op} {rhs}"
    # Fallback: stringify
    return str(p)

# NEW: Compose details using robust parsing for FILTER/JOIN and stable normalization for others
def _compose_details_with_eval_parsers(node):
    name = (node.get("name") or "").upper()
    info = node.get("extra_info", {}) or {}

    if "SCAN" in name:
        return f"from the table '{info.get('Table', 'unspecified')}'"

    if name == "PROJECTION":
        exprs = _ensure_list_and_get_exprs(info.get("Expressions"))
        return format_expression_list(exprs) if exprs else "the specified attributes"

    if name in ("FILTER",):
        raw = info.get("Filters")
        conjuncts = split_conjuncts(raw) if raw is not None else []
        preds = [parse_predicate(c) for c in conjuncts]
        if not preds:
            return "to apply the specified filters"
        text_parts = [_render_predicate_to_text(p) for p in preds]
        if len(text_parts) == 1:
            return f"to keep only records where {text_parts[0]}"
        return f"to keep only records where " + " and ".join(text_parts)

    if name in ("JOIN", "COMPARISON_JOIN", "ANY_JOIN", "DEPENDENT_JOIN", "CROSS_PRODUCT"):
        raw = info.get("Condition", info.get("Conditions"))
        conjuncts = split_conjuncts(raw) if raw is not None else []
        preds = [parse_predicate(c) for c in conjuncts]
        if not preds:
            return "based on the matching condition"
        text_parts = [_render_predicate_to_text(p) for p in preds]
        cond_text = " and ".join(text_parts)
        jt = info.get("JoinType")
        if jt:
            return f"using a {jt} join based on {cond_text}"
        return f"based on {cond_text}"

    if name == "AGGREGATE":
        aggs = _ensure_list_and_get_exprs(info.get("Expressions"))
        groups = _ensure_list_and_get_exprs(info.get("Groups"))
        aggs_txt = format_expression_list(aggs) if aggs else "the required aggregates"
        if groups:
            return f"{aggs_txt}, grouped by {format_expression_list(groups)}"
        return aggs_txt

    if name in ("SORT", "ORDER_BY"):
        orders = _ensure_list_and_get_exprs(info.get("Order"))
        pretty = _pretty_orders(orders)
        if pretty:
            txt = f"order the result by {_list_to_english(pretty)}"
        else:
            txt = "order the result"
        lim = info.get("Limit")
        if lim is not None and str(lim) != "":
            txt += f", keeping only the top {lim} record(s)"
        return txt

    if name == "LIMIT":
        return f"to the top {info.get('Limit', 'N')} records"

    if name == "DISTINCT":
        return "to remove duplicate records"

    if name in ("UNION", "INTERSECT", "EXCEPT"):
        return "based on the specified set operation"

    return "based on the specified parameters"

# --- NEW: Smooth English helpers ---

def _list_to_english(items, quoted=False):
    it = [f"'{x}'" if quoted else str(x) for x in items if str(x)]
    if not it: return ""
    if len(it) == 1: return it[0]
    return f"{', '.join(it[:-1])}, and {it[-1]}"

def _join_type_phrase(join_type):
    if not join_type: return "an inner join"
    jt = str(join_type).strip().lower().replace("_", " ")
    # Normalize common join type phrasings
    if jt in {"inner"}: return "an inner join"
    if jt in {"left", "left join"}: return "a left join"
    if jt in {"left outer", "left outer join"}: return "a left outer join"
    if jt in {"right", "right join"}: return "a right join"
    if jt in {"right outer", "right outer join"}: return "a right outer join"
    if jt in {"full", "full join"}: return "a full join"
    if jt in {"full outer", "full outer join"}: return "a full outer join"
    if jt in {"cross", "cross product", "cross join"}: return "a cross join"
    return f"a {jt} join"

def _pretty_orders(order_exprs):
    pretty = []
    for o in order_exprs:
        s = str(o).strip()
        m = re.search(r"\s+(asc|desc)\s*$", s, flags=re.IGNORECASE)
        if m:
            dir_ = m.group(1).lower()
            base = s[: m.start()].strip()
            pretty.append(f"{base} in {'ascending' if dir_=='asc' else 'descending'} order")
        else:
            pretty.append(s)
    return pretty

def _render_predicates_conj(raw):
    conjuncts = split_conjuncts(raw) if raw is not None else []
    preds = [parse_predicate(c) for c in conjuncts]
    if not preds:
        return ""
    parts = [_render_predicate_to_text(p) for p in preds]
    return " and ".join(parts)

def _compose_join_details_rich(node):
    info = node.get("extra_info", {}) or {}
    cond_text = _render_predicates_conj(info.get("Condition", info.get("Conditions")))
    jt_phrase = _join_type_phrase(info.get("JoinType"))

    # Try to name the sources from leaves
    left = node.get("children", [None, None])[0]
    right = node.get("children", [None, None])[1] if len(node.get("children", [])) > 1 else None
    left_tables = _leaf_tables_under(left) if left else []
    right_tables = _leaf_tables_under(right) if right else []
    lt = left_tables[0] if left_tables else "the left source"
    rt = right_tables[0] if right_tables else "the right source"

    if cond_text:
        return f"perform {jt_phrase} between {lt} and {rt} based on {cond_text}"
    return f"perform {jt_phrase} between {lt} and {rt}"

def _compose_filter_details_rich(node):
    info = node.get("extra_info", {}) or {}
    cond_text = _render_predicates_conj(info.get("Filters", info.get("Conditions")))
    if cond_text:
        return f"apply a filter to keep only records where {cond_text}"
    return "apply the necessary filters"

def _compose_projection_details_rich(node):
    info = node.get("extra_info", {}) or {}
    exprs = _ensure_list_and_get_exprs(info.get("Expressions"))
    if not exprs:
        return "select only the required attributes"
    return f"select the attributes {_list_to_english(exprs, quoted=True)}"

def _compose_aggregate_details_rich(node):
    info = node.get("extra_info", {}) or {}
    aggs = _ensure_list_and_get_exprs(info.get("Expressions"))
    groups = _ensure_list_and_get_exprs(info.get("Groups"))
    if aggs and groups:
        return f"group records by {_list_to_english(groups, quoted=True)} and compute {_list_to_english(aggs, quoted=True)}"
    if aggs:
        return f"compute {_list_to_english(aggs, quoted=True)} over the current records"
    if groups:
        return f"group records by {_list_to_english(groups, quoted=True)}"
    return "perform the required aggregations"

def _compose_sort_details_rich(node):
    info = node.get("extra_info", {}) or {}
    orders = _ensure_list_and_get_exprs(info.get("Order"))
    pretty = _pretty_orders(orders)
    if pretty:
        txt = f"order the result by {_list_to_english(pretty)}"
    else:
        txt = "order the result"
    lim = info.get("Limit")
    if lim is not None and str(lim) != "":
        txt += f", keeping only the top {lim} record(s)"
    return txt

def _compose_limit_details_rich(node):
    n = (node.get("extra_info", {}) or {}).get("Limit", "N")
    return f"restrict the output to the top {n} record(s)"

def _compose_setop_details_rich(node):
    name = (node.get("name") or "").upper()
    if name == "UNION":
        return "combine the two result sets with UNION, keeping unique rows"
    if name == "INTERSECT":
        return "retain only the rows common to both result sets (INTERSECT)"
    if name == "EXCEPT":
        return "take the rows from the first result set that do not appear in the second (EXCEPT)"
    return f"apply the set operation {name}"

def _compose_op_sentence(node, lead):
    op = (node.get("name") or "").upper()
    if "SCAN" in op:
        return None  # Avoid breaking the narrative flow with low-level scan mentions
    if op in ("JOIN", "COMPARISON_JOIN", "ANY_JOIN", "DEPENDENT_JOIN", "CROSS_PRODUCT"):
        return f"{lead}{_compose_join_details_rich(node)}."
    if op == "FILTER":
        return f"{lead}{_compose_filter_details_rich(node)}."
    if op == "PROJECTION":
        return f"{lead}{_compose_projection_details_rich(node)}."
    if op == "AGGREGATE":
        return f"{lead}{_compose_aggregate_details_rich(node)}."
    if op in ("SORT", "ORDER_BY"):
        return f"{lead}{_compose_sort_details_rich(node)}."
    if op == "LIMIT":
        return f"{lead}{_compose_limit_details_rich(node)}."
    if op == "DISTINCT":
        return f"{lead}remove duplicate rows so that each record appears only once."
    if op in ("UNION", "INTERSECT", "EXCEPT"):
        return f"{lead}{_compose_setop_details_rich(node)}."
    # Fallback to generic phrasing with improved detail parser
    return f"{lead}{_compose_details_with_eval_parsers(node)}."

# --- NEW: Smooth, enriched pre-order narrative ---

def serialize_pre_order_story(resolved_ra_tree):
    """
    Generates a cohesive, human-friendly story of the plan (top-down).
    - Mentions base tables up front
    - Uses robust parsing for conditions (FILTER/JOIN)
    - Enriches detail for PROJECTION/AGGREGATE/SORT/LIMIT/SETOPS
    - Avoids code-block style headings
    """
    sentences = []

    # 1) Introduce base tables
    base_tables = sorted(set(_leaf_tables_under(resolved_ra_tree)))
    if base_tables:
        if len(base_tables) == 1:
            sentences.append(f"The plan reads data from the base table {base_tables[0]}.")
        else:
            sentences.append(f"The plan reads data from the base tables {_list_to_english(base_tables)}.")

    # 2) Pre-order traversal with smooth connectors
    order = []
    def walk(n):
        order.append(n)
        for c in n.get("children", []) or []:
            walk(c)
    walk(resolved_ra_tree)

    step_idx = 0
    for n in order:
        s = None
        if step_idx == 0:
            s = _compose_op_sentence(n, "First, ")
        else:
            # Use a mix of connectors for smoother reading
            connector = "Then, " if step_idx == 1 else ("Next, " if step_idx < 5 else "After that, ")
            s = _compose_op_sentence(n, connector)
        if s:
            sentences.append(s)
            step_idx += 1

    # 3) Conclude if nothing but scans
    if step_idx == 0:
        sentences.append("The plan directly returns rows from the base table without additional processing.")

    return " ".join(sentences)

# --- 3. SERIALIZATION METHODS ---

def serialize_post_order_narrative(resolved_ra_tree):
    """
    (IMPROVED) METHOD 1: Generates a step-by-step narrative plan (bottom-up).
    Now correctly handles multiple parallel leaf nodes.
    """
    steps = []
    leaf_counter = 0
    
    def traverse(node):
        nonlocal leaf_counter
        is_leaf = not node.get("children")
        
        child_refs = [traverse(child) for child in node.get("children", [])]
        op_phrase, details = generate_smooth_nl_rephrase(node)
        
        # IMPROVED: Differentiate introductory phrases for parallel first steps
        intro_phrase = "Next,"
        if is_leaf:
            leaf_counter += 1
            intro_phrase = "First," if leaf_counter == 1 else "In a parallel first step,"
        
        narrative = f"{intro_phrase} {op_phrase} {details}."
        if child_refs:
            ref_str = " and ".join([f"the result of Step {ref}" for ref in child_refs])
            narrative += f" This is performed on {ref_str}."
            
        technical_context = f"Operation: {node['name']}"
        steps.append({'narrative': narrative, 'tech': technical_context})
        return len(steps)

    traverse(resolved_ra_tree)
    
    output = ["### Method 1: Post-order Narrative (Bottom-Up Reasoning Chain)\n"]
    for i, step in enumerate(steps):
        # output.append(f"**Step {i+1}:** {step['narrative']} *(Technical Op: {step['tech']})*")
        output.append(f"**Step {i+1}:** {step['narrative']} *({step['tech']})*")
    return "\n\n".join(output)

def serialize_pre_order_summary(resolved_ra_tree):
    """METHOD 2: Generates a high-level summary of the plan (top-down)."""
    summary_points = []

    def generate_summary(node, is_first=True):
        op_phrase, details = generate_smooth_nl_rephrase(node)
        point = f"The overall goal is to **{op_phrase}** {details}." if is_first else f"To do this, the plan must first **{op_phrase}** {details}."
        summary_points.append(point)
        for child in node.get("children", []):
            generate_summary(child, is_first=False)

    generate_summary(resolved_ra_tree)

    # output = "### Method 2: Pre-order Summary (Top-Down Strategy)\n\n"
    # Join with newlines for better readability
    output += "\n\n".join(summary_points)
    return output

def serialize_indented_narrative(resolved_ra_tree):
    """METHOD 3: Generates an indented, hierarchical outline of the plan."""
    lines = ["### Method 3: Indented Hierarchical Outline"]

    def traverse(node, level=0):
        indent = "  " * level
        op_phrase, details = generate_smooth_nl_rephrase(node)
        lines.append(f"{indent}- A **{node['name']}** operation is used to {op_phrase} {details}.")
        for child in node.get("children", []):
            traverse(child, level + 1)

    traverse(resolved_ra_tree)
    return "\n".join(lines)

def _leaf_tables_under(node):
    """Collect base table names under this subtree (from *SCAN nodes)."""
    tables = []
    def walk(n):
        name = (n.get("name") or "").upper()
        if "SCAN" in name:
            t = (n.get("extra_info") or {}).get("Table")
            if t and t not in tables:
                tables.append(t)
        for c in n.get("children", []) or []:
            walk(c)
    walk(node)
    return tables

#######################


def serialize_pre_order_summary_random(resolved_ra_tree, seed=None):
    """
    Generates a high-level summary of the plan (top-down) with randomized templates.
    
    Args:
        resolved_ra_tree: RA tree with pointers already resolved
        seed: Random seed for reproducibility (optional)
    
    Returns:
        str: Natural language summary with varied phrasing
    """
    if seed is not None:
        random.seed(seed)
    
    # Template variations for the root/first node
    FIRST_NODE_TEMPLATES = [
        "The overall goal is to **{op_phrase}** {details}.",
        "Our primary objective is to **{op_phrase}** {details}.",
        "The main goal is to **{op_phrase}** {details}.",
        "We aim to **{op_phrase}** {details}.",
        "The query seeks to **{op_phrase}** {details}.",
        "The plan's purpose is to **{op_phrase}** {details}.",
    ]
    
    # Template variations for subsequent nodes
    SUBSEQUENT_TEMPLATES = [
        "To do this, the plan must first **{op_phrase}** {details}.",
        "To achieve this, we need to **{op_phrase}** {details}.",
        "For this purpose, the plan must **{op_phrase}** {details}.",
        "To accomplish this, we first **{op_phrase}** {details}.",
        "This requires the plan to **{op_phrase}** {details}.",
        "To make this happen, we must **{op_phrase}** {details}.",
        "Before we can do that, the plan needs to **{op_phrase}** {details}.",
    ]
    
    summary_points = []

    def generate_summary(node, is_first=True):
        op_phrase, details = generate_smooth_nl_rephrase(node)
        
        # Select random template based on position
        if is_first:
            template = random.choice(FIRST_NODE_TEMPLATES)
        else:
            template = random.choice(SUBSEQUENT_TEMPLATES)
        
        point = template.format(op_phrase=op_phrase, details=details)
        summary_points.append(point)
        
        for child in node.get("children", []):
            generate_summary(child, is_first=False)

    generate_summary(resolved_ra_tree)
    
    # Join with newlines for better readability
    return "\n\n".join(summary_points)


#####################

import random

def serialize_random_plan_preorder(resolved_ra_tree, seed=None):
    """
    RECOMMENDED FOR TRAINING: Goal-oriented pre-order serialization with diverse language.
    
    Pre-order traversal with hierarchical "to do X, we need Y" structure.
    Uses randomized templates to prevent overfitting to rigid language patterns.
    
    Args:
        resolved_ra_tree: RA tree with pointers already resolved
        seed: Random seed for reproducibility (optional)
    
    Format designed for text-to-RA generation training:
    - Goal-oriented phrasing with template diversity
    - Pre-order traversal (parent before children)
    - Includes table scans
    - Hierarchical reasoning structure
    """
    
    if seed is not None:
        random.seed(seed)
    
    # Template variations for different depth levels
    ROOT_TEMPLATES = [
        "To answer this query, we need to **{operation}**.",
        "To solve this problem, we must **{operation}**.",
        "The query requires us to **{operation}**.",
        "Our goal is to **{operation}**.",
        "We begin by **{operation}**.",
        "First, we need to **{operation}**.",
        "The main objective is to **{operation}**.",
    ]
    
    DEPTH_1_TEMPLATES = [
        "To achieve this, we first **{operation}**.",
        "To accomplish this, we start by **{operation}**.",
        "For this purpose, we need to **{operation}**.",
        "To do this, we must **{operation}**.",
        "This requires us to **{operation}**.",
        "To reach this goal, we first **{operation}**.",
        "We accomplish this by **{operation}**.",
    ]
    
    DEEPER_TEMPLATES = [
        "Before that, we must **{operation}**.",
        "Prior to this, we need to **{operation}**.",
        "To support this, we first **{operation}**.",
        "As a prerequisite, we **{operation}**.",
        "Before we can do that, we **{operation}**.",
        "This step requires us to **{operation}**.",
        "Working backwards, we must **{operation}**.",
    ]
    
    output_lines = []
    
    def traverse_preorder(node, depth=0, is_first=True):
        name = (node.get("name") or "").upper()
        info = node.get("extra_info", {}) or {}
        
        # Skip VALUES nodes
        if name == "VALUES":
            for child in node.get("children", []):
                traverse_preorder(child, depth, False)
            return
        
        # Build operation description
        operation_desc = ""
        
        if "SCAN" in name:
            table = info.get("Table", "unknown")
            # Template variations for table access
            scan_templates = [
                f"access table '{table}'",
                f"retrieve data from table '{table}'",
                f"read from table '{table}'",
                f"scan table '{table}'",
                f"fetch records from table '{table}'",
            ]
            operation_desc = random.choice(scan_templates)
        
        elif name == "PROJECTION":
            exprs = _ensure_list_and_get_exprs(info.get("Expressions"))
            if not exprs:
                proj_templates = [
                    "select columns",
                    "project specific columns",
                    "choose the required columns",
                    "extract certain columns",
                ]
                operation_desc = random.choice(proj_templates)
            else:
                cleaned = [_clean_identifier(e) for e in exprs[:4]]
                if len(exprs) == 1:
                    proj_single_templates = [
                        f"select the column: {cleaned[0]}",
                        f"project column {cleaned[0]}",
                        f"extract the {cleaned[0]} column",
                        f"retrieve {cleaned[0]}",
                    ]
                    operation_desc = random.choice(proj_single_templates)
                elif len(exprs) <= 4:
                    cols = ", ".join(cleaned)
                    proj_multi_templates = [
                        f"select these columns: {cols}",
                        f"project columns: {cols}",
                        f"extract {cols}",
                        f"retrieve the columns {cols}",
                    ]
                    operation_desc = random.choice(proj_multi_templates)
                else:
                    cols = ", ".join(cleaned[:3])
                    operation_desc = f"select {len(exprs)} columns including: {cols}"
        
        elif name == "FILTER":
            conjuncts = _parse_filter_with_and_func(info.get("Filters"))
            if not conjuncts:
                filter_generic_templates = [
                    "filter rows to keep only relevant records",
                    "apply filters to retain necessary rows",
                    "keep only qualifying records",
                    "filter the data based on conditions",
                ]
                operation_desc = random.choice(filter_generic_templates)
            else:
                preds = [parse_predicate(c) for c in conjuncts[:3]]
                pred_strs = [_render_predicate_concise(p) for p in preds]
                
                if len(conjuncts) == 1:
                    filter_single_templates = [
                        f"filter rows where {pred_strs[0]}",
                        f"keep only rows where {pred_strs[0]}",
                        f"retain records where {pred_strs[0]}",
                        f"apply filter: {pred_strs[0]}",
                    ]
                    operation_desc = random.choice(filter_single_templates)
                elif len(conjuncts) <= 3:
                    conds = " AND ".join(pred_strs)
                    filter_multi_templates = [
                        f"filter rows where {conds}",
                        f"keep only rows where {conds}",
                        f"retain records satisfying {conds}",
                        f"apply filters: {conds}",
                    ]
                    operation_desc = random.choice(filter_multi_templates)
                else:
                    conds = " AND ".join(pred_strs[:2])
                    operation_desc = f"filter rows where {conds} AND {len(conjuncts)-2} more conditions"
        
        elif name in ("JOIN", "COMPARISON_JOIN", "ANY_JOIN", "DEPENDENT_JOIN"):
            join_type = info.get("JoinType", "INNER")
            conjuncts = split_conjuncts(info.get("Condition")) if info.get("Condition") else []
            
            if not conjuncts:
                join_generic_templates = [
                    f"perform {join_type} JOIN between the two data sources",
                    f"execute a {join_type} JOIN on the tables",
                    f"combine data using {join_type} JOIN",
                    f"join the two sources ({join_type})",
                ]
                operation_desc = random.choice(join_generic_templates)
            else:
                preds = [parse_predicate(c) for c in conjuncts[:2]]
                pred_strs = [_render_predicate_concise(p) for p in preds]
                conds = " AND ".join(pred_strs)
                
                join_cond_templates = [
                    f"perform {join_type} JOIN on {conds}",
                    f"join tables ({join_type}) where {conds}",
                    f"execute {join_type} JOIN matching {conds}",
                    f"combine data via {join_type} JOIN on {conds}",
                ]
                operation_desc = random.choice(join_cond_templates)
        
        elif name == "CROSS_PRODUCT":
            cross_templates = [
                "compute CROSS PRODUCT, combining every row from one source with every row from the other",
                "create Cartesian product of the two data sources",
                "generate all combinations via CROSS PRODUCT",
                "compute CROSS JOIN between the sources",
            ]
            operation_desc = random.choice(cross_templates)
        
        elif name == "AGGREGATE":
            agg_exprs = _ensure_list_and_get_exprs(info.get("Expressions"))
            group_exprs = _ensure_list_and_get_exprs(info.get("Groups"))
            
            # DISTINCT only
            if not agg_exprs or (len(agg_exprs) == 1 and str(agg_exprs[0]).upper().strip() == "DISTINCT"):
                if group_exprs:
                    groups = ", ".join([_clean_identifier(g) for g in group_exprs[:3]])
                    distinct_group_templates = [
                        f"compute distinct values grouped by {groups}",
                        f"find unique combinations of {groups}",
                        f"get distinct values per {groups}",
                        f"eliminate duplicates within {groups} groups",
                    ]
                    operation_desc = random.choice(distinct_group_templates)
                else:
                    distinct_templates = [
                        "compute distinct values, removing duplicate rows",
                        "eliminate duplicate records",
                        "find unique rows",
                        "remove duplicates from the dataset",
                    ]
                    operation_desc = random.choice(distinct_templates)
            else:
                # Regular aggregation
                agg_strs = []
                for expr in agg_exprs[:3]:
                    expr_str = str(expr)
                    if "COUNT(*)" in expr_str.upper():
                        agg_strs.append("COUNT(*)")
                    else:
                        cleaned = _clean_identifier(expr_str)
                        func_match = re.match(r"(\w+)\((.*)\)", cleaned, re.IGNORECASE)
                        if func_match:
                            func = func_match.group(1).upper()
                            arg = func_match.group(2)
                            agg_strs.append(f"{func}({arg})")
                        else:
                            agg_strs.append(cleaned)
                
                aggs = ", ".join(agg_strs)
                if group_exprs:
                    groups = ", ".join([_clean_identifier(g) for g in group_exprs[:3]])
                    agg_group_templates = [
                        f"aggregate {aggs} grouped by {groups}",
                        f"compute {aggs} for each {groups}",
                        f"calculate {aggs} per {groups} group",
                        f"group by {groups} and compute {aggs}",
                    ]
                    operation_desc = random.choice(agg_group_templates)
                else:
                    agg_templates = [
                        f"aggregate {aggs} across all rows",
                        f"compute {aggs} over the entire dataset",
                        f"calculate {aggs} globally",
                        f"compute overall {aggs}",
                    ]
                    operation_desc = random.choice(agg_templates)
        
        elif name in ("SORT", "ORDER_BY"):
            orders = _ensure_list_and_get_exprs(info.get("Order"))
            limit = info.get("Limit")
            offset = info.get("Offset")
            
            if not orders:
                sort_generic_templates = [
                    "sort results",
                    "order the output",
                    "arrange rows",
                    "organize the results",
                ]
                operation_desc = random.choice(sort_generic_templates)
            else:
                order_strs = []
                for o in orders[:2]:
                    o_str = str(o).strip()
                    desc_match = re.search(r"\s+(desc|asc)$", o_str, re.IGNORECASE)
                    if desc_match:
                        direction = desc_match.group(1).upper()
                        col = _clean_identifier(o_str[:desc_match.start()].strip())
                    else:
                        direction = "ASC"
                        col = _clean_identifier(o_str)
                    order_strs.append(f"{col} {direction}")
                
                order_clause = ", ".join(order_strs)
                if len(orders) > 2:
                    order_clause += f", and {len(orders)-2} more"
                
                sort_templates = [
                    f"sort results by {order_clause}",
                    f"order rows by {order_clause}",
                    f"arrange data by {order_clause}",
                    f"organize output by {order_clause}",
                ]
                operation_desc = random.choice(sort_templates)
                
                if offset and str(offset) != "0":
                    operation_desc += f", skipping first {offset}"
                if limit:
                    limit_phrases = [
                        f", limit to top {limit}",
                        f", keep only top {limit}",
                        f", return first {limit}",
                        f", restrict to {limit} rows",
                    ]
                    operation_desc += random.choice(limit_phrases)
        
        elif name == "LIMIT":
            limit = info.get("Limit", "N")
            limit_templates = [
                f"limit results to top {limit} rows",
                f"restrict output to {limit} records",
                f"return only the first {limit} rows",
                f"keep top {limit} results",
            ]
            operation_desc = random.choice(limit_templates)
        
        elif name == "DISTINCT":
            distinct_templates = [
                "remove duplicates, ensuring each row is unique",
                "eliminate duplicate rows",
                "keep only unique records",
                "ensure distinct results",
            ]
            operation_desc = random.choice(distinct_templates)
        
        elif name == "UNION":
            union_templates = [
                "combine result sets using UNION (keeping unique rows)",
                "merge the two result sets with UNION",
                "unite both result sets, removing duplicates",
                "perform UNION to combine unique rows",
            ]
            operation_desc = random.choice(union_templates)
        
        elif name == "INTERSECT":
            intersect_templates = [
                "intersect result sets, keeping only common rows",
                "find rows common to both result sets",
                "perform INTERSECT to get shared records",
                "retain only rows appearing in both sets",
            ]
            operation_desc = random.choice(intersect_templates)
        
        elif name == "EXCEPT":
            except_templates = [
                "compute set difference (rows in first set but not in second)",
                "perform EXCEPT to subtract second set from first",
                "remove rows from first set that appear in second",
                "find rows unique to the first result set",
            ]
            operation_desc = random.choice(except_templates)
        
        else:
            operation_desc = f"apply {name} to process the data"
        
        # Select template based on depth
        if is_first:
            template = random.choice(ROOT_TEMPLATES)
        elif depth == 1:
            template = random.choice(DEPTH_1_TEMPLATES)
        else:
            template = random.choice(DEEPER_TEMPLATES)
        
        line = template.format(operation=operation_desc)
        output_lines.append(line)
        
        # THEN process children (pre-order)
        for child in node.get("children", []):
            traverse_preorder(child, depth + 1, False)
    
    # Traverse in pre-order
    traverse_preorder(resolved_ra_tree, 0, True)
    
    return "\n".join(output_lines)

# def _format_cols_from_exprs(exprs):
#     cols = []
#     for e in to_list(exprs):
#         if isinstance(e, dict) and "expr" in e:
#             cols.append(e["expr"])
#         else:
#             cols.append(str(e))
#     return cols

# def _compose_join_step(node):
#     einfo = node.get("extra_info", {}) or {}
#     cond = einfo.get("Condition", einfo.get("Conditions"))
#     left = node.get("children", [None, None])[0]
#     right = node.get("children", [None, None])[1] if len(node.get("children", [])) > 1 else None
#     left_tables = _leaf_tables_under(left) if left else []
#     right_tables = _leaf_tables_under(right) if right else []
#     lt = left_tables[0] if left_tables else "the first source"
#     rt = right_tables[0] if right_tables else "the second source"
#     if isinstance(cond, dict):
#         cond_text = cond.get("expr", "a matching condition")
#     else:
#         cond_text = cond or "a matching condition"
#     return f"Join the {lt} and {rt} tables on {cond_text}"

# def _compose_filter_step(node):
#     einfo = node.get("extra_info", {}) or {}
#     cond = einfo.get("Filters", einfo.get("Conditions"))
#     if isinstance(cond, dict):
#         cond_text = cond.get("expr", "criteria are met")
#     else:
#         cond_text = cond or "criteria are met"
#     return f"Filter the result to include only records where {cond_text}"

# def _compose_projection_step(node):
#     einfo = node.get("extra_info", {}) or {}
#     cols = _format_cols_from_exprs(einfo.get("Expressions", []))
#     if not cols:
#         return "Project the necessary columns"
#     if len(cols) == 1:
#         return f"Project the column {cols[0]}"
#     return f"Project the columns {', '.join(cols[:-1])} and {cols[-1]}"

# def _compose_aggregate_step(node):
#     einfo = node.get("extra_info", {}) or {}
#     aggs = _format_cols_from_exprs(einfo.get("Expressions", []))
#     groups_raw = to_list(einfo.get("Groups", []))
#     groups = []
#     for g in groups_raw:
#         if isinstance(g, dict) and "expr" in g:
#             groups.append(g["expr"])
#         else:
#             groups.append(str(g))
#     aggs_txt = " and ".join(aggs) if aggs else "the required aggregates"
#     if groups:
#         if len(groups) == 1:
#             grp = groups[0]
#         else:
#             grp = f"{', '.join(groups[:-1])} and {groups[-1]}"
#         return f"Aggregate to compute {aggs_txt}, grouped by {grp}"
#     return f"Aggregate to compute {aggs_txt}"

# def _compose_sort_step(node):
#     einfo = node.get("extra_info", {}) or {}
#     order = to_list(einfo.get("Order", einfo.get("Order By", [])))
#     if not order:
#         return "Order the result"
#     # Detect DESC tokens
#     pretty = []
#     for item in order:
#         s = str(item)
#         if s.upper().endswith(" DESC"):
#             pretty.append(f"{s.rsplit(' ', 1)[0]} in descending order")
#         elif s.upper().endswith(" ASC"):
#             pretty.append(f"{s.rsplit(' ', 1)[0]} in ascending order")
#         else:
#             pretty.append(f"{s}")
#     if len(pretty) == 1:
#         return f"Order the result by {pretty[0]}"
#     return f"Order the result by {', '.join(pretty[:-1])} and {pretty[-1]}"

# def _compose_limit_step(node):
#     n = (node.get("extra_info", {}) or {}).get("Limit", "N")
#     return f"Limit the result to the top {n} record(s)"

# def _compose_distinct_step(node):
#     return "Remove duplicate rows (Distinct)"

# def _compose_setop_step(node, op):
#     op = op.upper()
#     if op == "UNION":
#         return "Union the two result sets, keeping unique rows"
#     if op == "INTERSECT":
#         return "Intersect the two result sets to keep only common rows"
#     if op == "EXCEPT":
#         return "Take the first result set and remove any rows that also appear in the second set (Except)"
#     return f"Apply the set operation {op}"

# def serialize_cot_style(resolved_ra_tree):
#     """
#     CoT-style serialization aligned with observed model distribution.
#     Produces:
#       - An intro line ("To solve this problem, we need to follow these steps:")
#       - A numbered list of concise steps (Join → Filter → Project → Aggregate → Order → Limit, etc.)
#       - An outro line ("Let's construct the relational algebra operator tree accordingly.")
#     """
#     # Post-order traversal to respect dataflow: children first, then parent
#     ordered_nodes = []
#     def walk(n):
#         for c in n.get("children", []) or []:
#             walk(c)
#         ordered_nodes.append(n)
#     walk(resolved_ra_tree)

#     # Build steps, skipping base scans
#     steps_raw = []
#     for n in ordered_nodes:
#         name = (n.get("name") or "").upper()
#         if "SCAN" in name:
#             continue
#         if name in ("ANY_JOIN", "COMPARISON_JOIN", "DEPENDENT_JOIN", "CROSS_PRODUCT", "JOIN"):
#             steps_raw.append(_compose_join_step(n))
#         elif name in ("FILTER",):
#             steps_raw.append(_compose_filter_step(n))
#         elif name in ("PROJECTION",):
#             steps_raw.append(_compose_projection_step(n))
#         elif name in ("AGGREGATE",):
#             steps_raw.append(_compose_aggregate_step(n))
#         elif name in ("SORT", "ORDER_BY"):
#             steps_raw.append(_compose_sort_step(n))
#         elif name in ("LIMIT",):
#             steps_raw.append(_compose_limit_step(n))
#         elif name in ("DISTINCT",):
#             steps_raw.append(_compose_distinct_step(n))
#         elif name in ("UNION", "INTERSECT", "EXCEPT"):
#             steps_raw.append(_compose_setop_step(n, name))
#         else:
#             # Fallback using existing NL helper
#             op_phrase, details = generate_smooth_nl_rephrase(n)
#             steps_raw.append(f"{op_phrase.capitalize()} {details}")

#     # Deduplicate adjacent identical steps to avoid noise
#     steps = []
#     for s in steps_raw:
#         if not steps or steps[-1] != s:
#             steps.append(s)

#     # Compose final CoT text
#     lines = ["To solve this problem, we need to follow these steps:"]
#     for i, s in enumerate(steps, 1):
#         lines.append(f"{i}. {s}.")
#     lines.append("Let's construct the relational algebra operator tree accordingly.")
#     return "\n".join(lines)

# """
# Goal-Oriented Pre-Order Serialization for Text-to-RA Training
# Maintains hierarchical reasoning structure while handling edge cases
# """

import os, sys, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation")))
from ra_eval_utils import (
    split_conjuncts,
    parse_predicate,
    _ensure_list_and_get_exprs,
)


def _clean_identifier(s):
    """Remove quotes and CAST from identifiers."""
    if not s:
        return s
    s = re.sub(r"^CAST\((.*)\)$", r"\1", s, flags=re.IGNORECASE)
    if len(s) >= 2 and s[0] in ("'", '"', "`") and s[-1] == s[0]:
        s = s[1:-1]
    return s


def _parse_filter_with_and_func(raw_filter):
    """Parse filter with AND() functional notation."""
    if not raw_filter:
        return []
    filter_str = raw_filter if isinstance(raw_filter, str) else raw_filter.get("expr", "")
    filter_str = str(filter_str).strip()
    
    and_func_match = re.match(r"^AND\((.*)\)$", filter_str, flags=re.IGNORECASE)
    if and_func_match:
        inner = and_func_match.group(1)
        parts = []
        depth, current = 0, []
        for char in inner:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            parts.append(''.join(current).strip())
        return parts
    return split_conjuncts(filter_str)


def _render_predicate_concise(p):
    """Render predicate in concise, pattern-learnable form."""
    kind = p.get("kind")
    
    if kind in ("is_null", "is_not_null"):
        lhs = _clean_identifier(p.get("lhs_expr") or p.get("lhs") or "field")
        return f"{lhs} is {'not ' if kind == 'is_not_null' else ''}NULL"
    
    if kind == "in":
        lhs = _clean_identifier(p.get("lhs") or "field")
        rhs_list = p.get("rhs_list") or []
        vals = ", ".join([_clean_identifier(str(x)) for x in rhs_list[:3]])
        suffix = f" (and {len(rhs_list)-3} more)" if len(rhs_list) > 3 else ""
        return f"{lhs} IN ({vals}{suffix})"
    
    if kind == "not_in":
        lhs = _clean_identifier(p.get("lhs") or "field")
        rhs_list = p.get("rhs_list") or []
        vals = ", ".join([_clean_identifier(str(x)) for x in rhs_list[:3]])
        return f"{lhs} NOT IN ({vals})"
    
    if kind == "between":
        lhs = _clean_identifier(p.get("lhs") or "field")
        lo, hi = p.get("range") or (None, None)
        return f"{lhs} BETWEEN {_clean_identifier(str(lo))} AND {_clean_identifier(str(hi))}"
    
    if kind == "like":
        lhs = _clean_identifier(p.get("lhs") or "field")
        rhs = p.get("rhs")
        return f"{lhs} LIKE {rhs}"
    
    if kind == "cmp":
        lhs = _clean_identifier(p.get("lhs") or "field")
        op = p.get("op") or "="
        rhs = _clean_identifier(str(p.get("rhs"))) if p.get("rhs") else "value"
        # Handle subqueries
        if "$SCALAR_QUERY()" in str(p.get("rhs")):
            rhs = "(subquery)"
        return f"{lhs} {op} {rhs}"
    
    return str(p)


def _generate_operation_phrase(node):
    """
    Generate operation phrase and details for goal-oriented reasoning.
    Returns: (action_verb, purpose_details)
    """
    name = (node.get("name") or "").upper()
    info = node.get("extra_info", {}) or {}
    
    # PROJECTION
    if name == "PROJECTION":
        exprs = _ensure_list_and_get_exprs(info.get("Expressions"))
        if not exprs:
            return "select columns", "to retrieve the required attributes"
        
        # Clean expressions
        cleaned = [_clean_identifier(e) for e in exprs[:4]]
        if len(cleaned) == 1:
            return "select", f"the column: {cleaned[0]}"
        elif len(exprs) <= 4:
            cols = ", ".join(cleaned)
            return "select", f"these columns: {cols}"
        else:
            cols = ", ".join(cleaned[:3])
            return "select", f"{len(exprs)} columns including: {cols}"
    
    # FILTER
    elif name == "FILTER":
        conjuncts = _parse_filter_with_and_func(info.get("Filters"))
        if not conjuncts:
            return "filter rows", "to keep only relevant records"
        
        preds = [parse_predicate(c) for c in conjuncts]
        pred_strs = [_render_predicate_concise(p) for p in preds[:3]]
        
        if len(pred_strs) == 1:
            return "filter rows", f"where {pred_strs[0]}"
        elif len(pred_strs) <= 3:
            conds = " AND ".join(pred_strs)
            return "filter rows", f"where {conds}"
        else:
            conds = " AND ".join(pred_strs[:2])
            return "filter rows", f"where {conds} (and {len(pred_strs)-2} more conditions)"
    
    # JOIN
    elif name in ("JOIN", "COMPARISON_JOIN", "ANY_JOIN", "DEPENDENT_JOIN"):
        join_type = info.get("JoinType", "INNER")
        conjuncts = split_conjuncts(info.get("Condition")) if info.get("Condition") else []
        
        if not conjuncts:
            return f"perform {join_type} JOIN", "between the two data sources"
        
        preds = [parse_predicate(c) for c in conjuncts[:2]]
        pred_strs = [_render_predicate_concise(p) for p in preds]
        
        if len(pred_strs) == 1:
            return f"perform {join_type} JOIN", f"on {pred_strs[0]}"
        else:
            conds = " AND ".join(pred_strs)
            return f"perform {join_type} JOIN", f"on {conds}"
    
    # CROSS_PRODUCT
    elif name == "CROSS_PRODUCT":
        return "compute CROSS PRODUCT", "combining every row from one source with every row from the other"
    
    # AGGREGATE
    elif name == "AGGREGATE":
        agg_exprs = _ensure_list_and_get_exprs(info.get("Expressions"))
        group_exprs = _ensure_list_and_get_exprs(info.get("Groups"))
        
        # DISTINCT only
        if not agg_exprs or (len(agg_exprs) == 1 and str(agg_exprs[0]).upper().strip() == "DISTINCT"):
            if group_exprs:
                groups = ", ".join([_clean_identifier(g) for g in group_exprs[:3]])
                return "compute distinct values", f"grouped by {groups}"
            return "compute distinct values", "removing duplicate rows"
        
        # Parse aggregations
        agg_strs = []
        for expr in agg_exprs[:3]:
            expr_clean = _clean_identifier(str(expr))
            if "COUNT(*)" in str(expr).upper():
                agg_strs.append("COUNT(*)")
            elif "COUNT" in str(expr).upper():
                agg_strs.append(f"COUNT({expr_clean})")
            elif "SUM" in str(expr).upper():
                agg_strs.append(f"SUM({expr_clean})")
            elif "AVG" in str(expr).upper():
                agg_strs.append(f"AVG({expr_clean})")
            elif "MAX" in str(expr).upper():
                agg_strs.append(f"MAX({expr_clean})")
            elif "MIN" in str(expr).upper():
                agg_strs.append(f"MIN({expr_clean})")
            else:
                agg_strs.append(expr_clean)
        
        aggs = ", ".join(agg_strs)
        
        if group_exprs:
            groups = ", ".join([_clean_identifier(g) for g in group_exprs[:3]])
            return "aggregate", f"{aggs} grouped by {groups}"
        return "aggregate", f"{aggs} across all rows"
    
    # SORT
    elif name in ("SORT", "ORDER_BY"):
        orders = _ensure_list_and_get_exprs(info.get("Order"))
        limit = info.get("Limit")
        offset = info.get("Offset")
        
        order_strs = []
        for o in orders[:2]:
            o_str = str(o).strip()
            desc_match = re.search(r"\s+(desc|asc)$", o_str, re.IGNORECASE)
            if desc_match:
                direction = desc_match.group(1).upper()
                col = _clean_identifier(o_str[:desc_match.start()].strip())
            else:
                direction = "ASC"
                col = _clean_identifier(o_str)
            order_strs.append(f"{col} {direction}")
        
        order_clause = ", ".join(order_strs)
        if len(orders) > 2:
            order_clause += f" (and {len(orders)-2} more)"
        
        details = f"by {order_clause}"
        if offset and str(offset) != "0":
            details += f", skip first {offset}"
        if limit:
            details += f", limit to top {limit}"
        
        return "sort results", details
    
    # LIMIT
    elif name == "LIMIT":
        limit = info.get("Limit", "N")
        return "limit results", f"to top {limit} rows"
    
    # DISTINCT
    elif name == "DISTINCT":
        return "remove duplicates", "ensuring each row is unique"
    
    # SET OPERATIONS
    elif name == "UNION":
        return "combine result sets", "using UNION (keeping unique rows)"
    elif name == "INTERSECT":
        return "intersect result sets", "keeping only common rows"
    elif name == "EXCEPT":
        return "compute set difference", "rows in first set but not in second"
    
    # Default
    else:
        return f"apply {name}", "to process the data"


def serialize_goal_oriented_preorder(resolved_ra_tree):
    """
    Goal-oriented pre-order serialization for text-to-RA training.
    Maintains "goal → subgoal" structure while handling edge cases.
    """
    
    summary_points = []
    
    def generate_reasoning(node, depth=0, is_first=True):
        name = (node.get("name") or "").upper()
        
        # Skip leaf scans and VALUES
        if "SCAN" in name or name == "VALUES":
            for child in node.get("children", []):
                generate_reasoning(child, depth, False)
            return
        
        # Generate operation phrase
        action, details = _generate_operation_phrase(node)
        
        # Format based on depth for hierarchical reasoning
        if is_first:
            point = f"To answer this query, we need to **{action}** {details}."
        elif depth == 1:
            point = f"To achieve this, we first **{action}** {details}."
        else:
            point = f"Before that, we must **{action}** {details}."
        
        summary_points.append(point)
        
        # Recursively process children
        for child in node.get("children", []):
            generate_reasoning(child, depth + 1, False)
    
    generate_reasoning(resolved_ra_tree, 0, True)
    
    return "\n".join(summary_points)


def serialize_step_by_step_reasoning(resolved_ra_tree):
    """
    Alternative: Step-by-step numbered format (more explicit for training).
    """
    
    steps = []
    
    def collect_steps(node):
        name = (node.get("name") or "").upper()
        
        if "SCAN" in name or name == "VALUES":
            for child in node.get("children", []):
                collect_steps(child)
            return
        
        action, details = _generate_operation_phrase(node)
        steps.append(f"**{action}** {details}")
        
        for child in node.get("children", []):
            collect_steps(child)
    
    collect_steps(resolved_ra_tree)
    
    output = ["**Reasoning Steps:**\n"]
    for i, step in enumerate(steps, 1):
        output.append(f"{i}. {step}")
    
    return "\n".join(output)


def serialize_why_chain(resolved_ra_tree):
    """
    Alternative: Explicit "WHY" reasoning chain.
    Best for teaching the model the purpose of each operation.
    """
    
    chain = []
    
    def build_chain(node, parent_purpose=None):
        name = (node.get("name") or "").upper()
        
        if "SCAN" in name or name == "VALUES":
            for child in node.get("children", []):
                build_chain(child, parent_purpose)
            return
        
        action, details = _generate_operation_phrase(node)
        
        if parent_purpose:
            entry = f"WHY {action}? → To {parent_purpose}\nHOW? → {details}"
        else:
            entry = f"GOAL: {action} {details}"
        
        chain.append(entry)
        
        # Pass this operation's purpose to children
        child_purpose = f"{action} {details}".lower()
        for child in node.get("children", []):
            build_chain(child, child_purpose)
    
    build_chain(resolved_ra_tree)
    
    return "\n\n".join(chain)


def serialize_complete_execution_plan(resolved_ra_tree):
    """
    RECOMMENDED FOR TRAINING: Complete execution plan with numbered steps.
    
    Includes ALL operations (SEQ_SCAN, FILTER, JOIN, etc.) in clear format.
    Shows explicit data flow and table access patterns.
    
    Format designed for text-to-RA generation training:
    - Numbered steps for clear ordering
    - Includes table scans (critical for multi-table queries)
    - Consistent pattern-learnable structure
    - Concise but complete
    """
    
    steps = []
    step_counter = [1]  # Use list to allow modification in nested function
    
    def traverse_and_record(node):
        name = (node.get("name") or "").upper()
        info = node.get("extra_info", {}) or {}
        
        # Handle SEQ_SCAN - CRITICAL for training
        if "SCAN" in name:
            table = info.get("Table", "unknown")
            steps.append(f"Step {step_counter[0]}: [SCAN] Access table '{table}'")
            step_counter[0] += 1
            return
        
        # Skip VALUES nodes
        if name == "VALUES":
            return
        
        # Process children first (pre-order but with scans first)
        for child in node.get("children", []):
            traverse_and_record(child)
        
        # Then process current node
        operation_str = ""
        
        if name == "PROJECTION":
            exprs = _ensure_list_and_get_exprs(info.get("Expressions"))
            if not exprs:
                operation_str = "[PROJECT] Select columns"
            else:
                cleaned = [_clean_identifier(e) for e in exprs[:4]]
                if len(exprs) == 1:
                    operation_str = f"[PROJECT] Select: {cleaned[0]}"
                elif len(exprs) <= 4:
                    cols = ", ".join(cleaned)
                    operation_str = f"[PROJECT] Select: {cols}"
                else:
                    cols = ", ".join(cleaned[:3])
                    operation_str = f"[PROJECT] Select {len(exprs)} columns: {cols}, ..."
        
        elif name == "FILTER":
            conjuncts = _parse_filter_with_and_func(info.get("Filters"))
            if not conjuncts:
                operation_str = "[FILTER] Apply filter conditions"
            else:
                preds = [parse_predicate(c) for c in conjuncts[:3]]
                pred_strs = [_render_predicate_concise(p) for p in preds]
                if len(conjuncts) == 1:
                    operation_str = f"[FILTER] Where {pred_strs[0]}"
                elif len(conjuncts) <= 3:
                    conds = " AND ".join(pred_strs)
                    operation_str = f"[FILTER] Where {conds}"
                else:
                    conds = " AND ".join(pred_strs[:2])
                    operation_str = f"[FILTER] Where {conds} AND {len(conjuncts)-2} more"
        
        elif name in ("JOIN", "COMPARISON_JOIN", "ANY_JOIN", "DEPENDENT_JOIN"):
            join_type = info.get("JoinType", "INNER")
            conjuncts = split_conjuncts(info.get("Condition")) if info.get("Condition") else []
            
            if not conjuncts:
                operation_str = f"[JOIN] {join_type} JOIN on (implicit)"
            else:
                preds = [parse_predicate(c) for c in conjuncts[:2]]
                pred_strs = [_render_predicate_concise(p) for p in preds]
                conds = " AND ".join(pred_strs)
                operation_str = f"[JOIN] {join_type} JOIN on {conds}"
        
        elif name == "CROSS_PRODUCT":
            operation_str = "[JOIN] CROSS PRODUCT"
        
        elif name == "AGGREGATE":
            agg_exprs = _ensure_list_and_get_exprs(info.get("Expressions"))
            group_exprs = _ensure_list_and_get_exprs(info.get("Groups"))
            
            # DISTINCT only
            if not agg_exprs or (len(agg_exprs) == 1 and str(agg_exprs[0]).upper().strip() == "DISTINCT"):
                if group_exprs:
                    groups = ", ".join([_clean_identifier(g) for g in group_exprs[:3]])
                    operation_str = f"[AGGREGATE] DISTINCT grouped by {groups}"
                else:
                    operation_str = "[AGGREGATE] DISTINCT"
            else:
                # Regular aggregation
                agg_strs = []
                for expr in agg_exprs[:3]:
                    expr_str = str(expr)
                    if "COUNT(*)" in expr_str.upper():
                        agg_strs.append("COUNT(*)")
                    else:
                        cleaned = _clean_identifier(expr_str)
                        # Extract function name
                        func_match = re.match(r"(\w+)\((.*)\)", cleaned, re.IGNORECASE)
                        if func_match:
                            func = func_match.group(1).upper()
                            arg = func_match.group(2)
                            agg_strs.append(f"{func}({arg})")
                        else:
                            agg_strs.append(cleaned)
                
                aggs = ", ".join(agg_strs)
                if group_exprs:
                    groups = ", ".join([_clean_identifier(g) for g in group_exprs[:3]])
                    operation_str = f"[AGGREGATE] {aggs} GROUP BY {groups}"
                else:
                    operation_str = f"[AGGREGATE] {aggs}"
        
        elif name in ("SORT", "ORDER_BY"):
            orders = _ensure_list_and_get_exprs(info.get("Order"))
            limit = info.get("Limit")
            offset = info.get("Offset")
            
            if not orders:
                operation_str = "[SORT] Sort results"
            else:
                order_strs = []
                for o in orders[:2]:
                    o_str = str(o).strip()
                    desc_match = re.search(r"\s+(desc|asc)$", o_str, re.IGNORECASE)
                    if desc_match:
                        direction = desc_match.group(1).upper()
                        col = _clean_identifier(o_str[:desc_match.start()].strip())
                    else:
                        direction = "ASC"
                        col = _clean_identifier(o_str)
                    order_strs.append(f"{col} {direction}")
                
                order_clause = ", ".join(order_strs)
                if len(orders) > 2:
                    order_clause += f", ..."
                
                operation_str = f"[SORT] ORDER BY {order_clause}"
                
                if limit:
                    operation_str += f" LIMIT {limit}"
                if offset and str(offset) != "0":
                    operation_str += f" OFFSET {offset}"
        
        elif name == "LIMIT":
            limit = info.get("Limit", "N")
            operation_str = f"[LIMIT] Return top {limit} rows"
        
        elif name == "DISTINCT":
            operation_str = "[DISTINCT] Remove duplicates"
        
        elif name == "UNION":
            operation_str = "[UNION] Combine result sets (unique rows)"
        
        elif name == "INTERSECT":
            operation_str = "[INTERSECT] Keep common rows"
        
        elif name == "EXCEPT":
            operation_str = "[EXCEPT] Set difference"
        
        else:
            operation_str = f"[{name}] Apply operation"
        
        if operation_str:
            steps.append(f"Step {step_counter[0]}: {operation_str}")
            step_counter[0] += 1
    
    traverse_and_record(resolved_ra_tree)
    
    # Build output
    output = ["=== Query Execution Plan ===\n"]
    output.extend(steps)
    output.append("\n=== End of Plan ===")
    
    return "\n".join(output)

def serialize_complete_execution_plan_preorder(resolved_ra_tree):
    """
    RECOMMENDED FOR TRAINING: Goal-oriented pre-order serialization.
    
    Pre-order traversal with hierarchical "to do X, we need Y" structure.
    Includes ALL operations in the order they appear in the tree structure.
    
    Format designed for text-to-RA generation training:
    - Goal-oriented phrasing (to answer this query, to achieve this, etc.)
    - Pre-order traversal (parent before children)
    - Includes table scans
    - Hierarchical reasoning structure
    """
    
    output_lines = []
    
    def traverse_preorder(node, depth=0, is_first=True):
        name = (node.get("name") or "").upper()
        info = node.get("extra_info", {}) or {}
        
        # Skip VALUES nodes
        if name == "VALUES":
            for child in node.get("children", []):
                traverse_preorder(child, depth, False)
            return
        
        # Build operation description
        operation_desc = ""
        
        if "SCAN" in name:
            table = info.get("Table", "unknown")
            operation_desc = f"access table '{table}'"
        
        elif name == "PROJECTION":
            exprs = _ensure_list_and_get_exprs(info.get("Expressions"))
            if not exprs:
                operation_desc = "select columns"
            else:
                cleaned = [_clean_identifier(e) for e in exprs[:4]]
                if len(exprs) == 1:
                    operation_desc = f"select the column: {cleaned[0]}"
                elif len(exprs) <= 4:
                    cols = ", ".join(cleaned)
                    operation_desc = f"select these columns: {cols}"
                else:
                    cols = ", ".join(cleaned[:3])
                    operation_desc = f"select {len(exprs)} columns including: {cols}"
        
        elif name == "FILTER":
            conjuncts = _parse_filter_with_and_func(info.get("Filters"))
            if not conjuncts:
                operation_desc = "filter rows to keep only relevant records"
            else:
                preds = [parse_predicate(c) for c in conjuncts[:3]]
                pred_strs = [_render_predicate_concise(p) for p in preds]
                if len(conjuncts) == 1:
                    operation_desc = f"filter rows where {pred_strs[0]}"
                elif len(conjuncts) <= 3:
                    conds = " AND ".join(pred_strs)
                    operation_desc = f"filter rows where {conds}"
                else:
                    conds = " AND ".join(pred_strs[:2])
                    operation_desc = f"filter rows where {conds} AND {len(conjuncts)-2} more conditions"
        
        elif name in ("JOIN", "COMPARISON_JOIN", "ANY_JOIN", "DEPENDENT_JOIN"):
            join_type = info.get("JoinType", "INNER")
            conjuncts = split_conjuncts(info.get("Condition")) if info.get("Condition") else []
            
            if not conjuncts:
                operation_desc = f"perform {join_type} JOIN between the two data sources"
            else:
                preds = [parse_predicate(c) for c in conjuncts[:2]]
                pred_strs = [_render_predicate_concise(p) for p in preds]
                conds = " AND ".join(pred_strs)
                operation_desc = f"perform {join_type} JOIN on {conds}"
        
        elif name == "CROSS_PRODUCT":
            operation_desc = "compute CROSS PRODUCT, combining every row from one source with every row from the other"
        
        elif name == "AGGREGATE":
            agg_exprs = _ensure_list_and_get_exprs(info.get("Expressions"))
            group_exprs = _ensure_list_and_get_exprs(info.get("Groups"))
            
            # DISTINCT only
            if not agg_exprs or (len(agg_exprs) == 1 and str(agg_exprs[0]).upper().strip() == "DISTINCT"):
                if group_exprs:
                    groups = ", ".join([_clean_identifier(g) for g in group_exprs[:3]])
                    operation_desc = f"compute distinct values grouped by {groups}"
                else:
                    operation_desc = "compute distinct values, removing duplicate rows"
            else:
                # Regular aggregation
                agg_strs = []
                for expr in agg_exprs[:3]:
                    expr_str = str(expr)
                    if "COUNT(*)" in expr_str.upper():
                        agg_strs.append("COUNT(*)")
                    else:
                        cleaned = _clean_identifier(expr_str)
                        func_match = re.match(r"(\w+)\((.*)\)", cleaned, re.IGNORECASE)
                        if func_match:
                            func = func_match.group(1).upper()
                            arg = func_match.group(2)
                            agg_strs.append(f"{func}({arg})")
                        else:
                            agg_strs.append(cleaned)
                
                aggs = ", ".join(agg_strs)
                if group_exprs:
                    groups = ", ".join([_clean_identifier(g) for g in group_exprs[:3]])
                    operation_desc = f"aggregate {aggs} grouped by {groups}"
                else:
                    operation_desc = f"aggregate {aggs} across all rows"
        
        elif name in ("SORT", "ORDER_BY"):
            orders = _ensure_list_and_get_exprs(info.get("Order"))
            limit = info.get("Limit")
            offset = info.get("Offset")
            
            if not orders:
                operation_desc = "sort results"
            else:
                order_strs = []
                for o in orders[:2]:
                    o_str = str(o).strip()
                    desc_match = re.search(r"\s+(desc|asc)$", o_str, re.IGNORECASE)
                    if desc_match:
                        direction = desc_match.group(1).upper()
                        col = _clean_identifier(o_str[:desc_match.start()].strip())
                    else:
                        direction = "ASC"
                        col = _clean_identifier(o_str)
                    order_strs.append(f"{col} {direction}")
                
                order_clause = ", ".join(order_strs)
                if len(orders) > 2:
                    order_clause += f", and {len(orders)-2} more"
                
                operation_desc = f"sort results by {order_clause}"
                
                if offset and str(offset) != "0":
                    operation_desc += f", skip first {offset}"
                if limit:
                    operation_desc += f", limit to top {limit}"
        
        elif name == "LIMIT":
            limit = info.get("Limit", "N")
            operation_desc = f"limit results to top {limit} rows"
        
        elif name == "DISTINCT":
            operation_desc = "remove duplicates, ensuring each row is unique"
        
        elif name == "UNION":
            operation_desc = "combine result sets using UNION (keeping unique rows)"
        
        elif name == "INTERSECT":
            operation_desc = "intersect result sets, keeping only common rows"
        
        elif name == "EXCEPT":
            operation_desc = "compute set difference (rows in first set but not in second)"
        
        else:
            operation_desc = f"apply {name} to process the data"
        
        # Format based on depth for hierarchical reasoning
        if is_first:
            line = f"To answer this query, we need to **{operation_desc}**."
        elif depth == 1:
            line = f"To achieve this, we first **{operation_desc}**."
        else:
            line = f"Before that, we must **{operation_desc}**."
        
        output_lines.append(line)
        
        # THEN process children (pre-order)
        for child in node.get("children", []):
            traverse_preorder(child, depth + 1, False)
    
    # Traverse in pre-order
    traverse_preorder(resolved_ra_tree, 0, True)
    
    return "\n".join(output_lines)

# """
# Enhanced Pre-Order Serialization for Relational Algebra Trees
# Handles all corner cases including quoted identifiers, CAST, AND(), subqueries, etc.
# """

# import os, sys, re
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from eval.ra_eval_utils import (
#     split_conjuncts,
#     parse_predicate,
#     tokenize_ra_expression,
#     _ensure_list_and_get_exprs,
# )


# def _clean_identifier(s):
#     """Remove quotes and CAST from identifiers for readability."""
#     if not s:
#         return s
#     # Remove CAST() wrapper
#     s = re.sub(r"^CAST\((.*)\)$", r"\1", s, flags=re.IGNORECASE)
#     # Remove quotes
#     if len(s) >= 2 and s[0] in ("'", '"', "`") and s[-1] == s[0]:
#         s = s[1:-1]
#     return s


# def _explain_expression(expr_str):
#     """Explain a complex expression in natural language."""
#     if not expr_str:
#         return "an expression"
    
#     expr_str = str(expr_str).strip()
    
#     # Handle subquery placeholders
#     if "$SCALAR_QUERY()" in expr_str:
#         expr_str = expr_str.replace("$SCALAR_QUERY()", "a subquery result")
    
#     # Detect arithmetic operations
#     if "/" in expr_str and ("*" in expr_str or "+" in expr_str or "-" in expr_str):
#         return f"the computed value from: {expr_str}"
#     elif "/" in expr_str:
#         return f"the ratio: {expr_str}"
#     elif "*" in expr_str:
#         return f"the product: {expr_str}"
#     elif "+" in expr_str and "-" in expr_str:
#         return f"the calculation: {expr_str}"
    
#     # Detect SQL functions
#     func_match = re.match(r"(\w+)\((.*)\)", expr_str, re.IGNORECASE)
#     if func_match:
#         func_name = func_match.group(1).upper()
#         func_arg = func_match.group(2)
        
#         if func_name in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
#             return f"{func_name.lower()} of {_clean_identifier(func_arg)}"
#         elif func_name == "SUBSTRING":
#             return f"substring extracted from {expr_str}"
#         elif func_name == "ROUND":
#             return f"rounded value from {expr_str}"
#         elif func_name == "CAST":
#             return _explain_expression(func_arg)
#         elif func_name == "CASE":
#             return f"conditional value based on: {expr_str}"
#         elif func_name == "RANK":
#             return "row ranking"
#         elif func_name == "ABS":
#             return f"absolute value of {func_arg}"
#         else:
#             return f"{func_name}({func_arg})"
    
#     # Clean and return
#     return _clean_identifier(expr_str)


# def _render_predicate_detailed(p):
#     """Render a parsed predicate into detailed, natural English."""
#     kind = p.get("kind")
    
#     if kind == "is_null":
#         lhs = p.get("lhs_expr") or p.get("lhs") or "the field"
#         lhs_clean = _clean_identifier(lhs)
#         return f"{lhs_clean} is NULL (missing data)"
    
#     if kind == "is_not_null":
#         lhs = p.get("lhs_expr") or p.get("lhs") or "the field"
#         lhs_clean = _clean_identifier(lhs)
#         return f"{lhs_clean} is present (not NULL)"
    
#     if kind == "in":
#         lhs = _clean_identifier(p.get("lhs") or "the field")
#         rhs_list = p.get("rhs_list") or []
#         cleaned_vals = [_clean_identifier(str(x)) for x in rhs_list]
#         if len(cleaned_vals) <= 3:
#             vals = ", ".join(cleaned_vals)
#             return f"{lhs} matches one of: {vals}"
#         else:
#             vals = ", ".join(cleaned_vals[:3])
#             return f"{lhs} matches one of {len(cleaned_vals)} values (e.g., {vals}...)"
    
#     if kind == "not_in":
#         lhs = _clean_identifier(p.get("lhs") or "the field")
#         rhs_list = p.get("rhs_list") or []
#         cleaned_vals = [_clean_identifier(str(x)) for x in rhs_list]
#         if len(cleaned_vals) <= 3:
#             vals = ", ".join(cleaned_vals)
#             return f"{lhs} does NOT match any of: {vals}"
#         else:
#             return f"{lhs} excludes {len(cleaned_vals)} specific values"
    
#     if kind == "between":
#         lhs = _clean_identifier(p.get("lhs") or "the field")
#         lo, hi = p.get("range") or (None, None)
#         lo_clean = _clean_identifier(str(lo)) if lo else "?"
#         hi_clean = _clean_identifier(str(hi)) if hi else "?"
#         return f"{lhs} is between {lo_clean} and {hi_clean} (inclusive)"
    
#     if kind == "like":
#         lhs = _clean_identifier(p.get("lhs") or "the field")
#         rhs = p.get("rhs")
#         return f"{lhs} matches the pattern {rhs}"
    
#     if kind == "cmp":
#         lhs = p.get("lhs") or "the field"
#         op = p.get("op") or "="
#         rhs = p.get("rhs")
        
#         # Clean identifiers
#         lhs_clean = _explain_expression(lhs)
#         rhs_clean = _explain_expression(str(rhs)) if rhs else "a value"
        
#         # Map operators to natural language
#         op_map = {
#             "=": "equals",
#             "==": "equals",
#             "!=": "is not equal to",
#             "<>": "is not equal to",
#             "<": "is less than",
#             "<=": "is less than or equal to",
#             ">": "is greater than",
#             ">=": "is greater than or equal to"
#         }
#         op_text = op_map.get(op, op)
        
#         return f"{lhs_clean} {op_text} {rhs_clean}"
    
#     return str(p)


# def _parse_filter_with_and_func(raw_filter):
#     """Parse filter that might use AND() functional notation."""
#     if not raw_filter:
#         return []
    
#     filter_str = raw_filter
#     if isinstance(raw_filter, dict):
#         filter_str = raw_filter.get("expr", "")
    
#     filter_str = str(filter_str).strip()
    
#     # Check for AND() functional notation
#     and_func_match = re.match(r"^AND\((.*)\)$", filter_str, flags=re.IGNORECASE)
#     if and_func_match:
#         # Extract arguments from AND(arg1, arg2, ...)
#         inner = and_func_match.group(1)
#         # Simple split by comma (not perfect but works for most cases)
#         parts = []
#         depth = 0
#         current = []
#         for char in inner:
#             if char == '(':
#                 depth += 1
#             elif char == ')':
#                 depth -= 1
#             elif char == ',' and depth == 0:
#                 parts.append(''.join(current).strip())
#                 current = []
#                 continue
#             current.append(char)
#         if current:
#             parts.append(''.join(current).strip())
#         return parts
    
#     # Otherwise use split_conjuncts
#     return split_conjuncts(filter_str)


# def _explain_filter_conditions(raw_filter):
#     """Generate detailed explanation of filter conditions."""
#     if not raw_filter:
#         return "applies some filtering criteria"
    
#     conjuncts = _parse_filter_with_and_func(raw_filter)
#     if not conjuncts:
#         return "applies filtering criteria"
    
#     predicates = [parse_predicate(c) for c in conjuncts]
    
#     if len(predicates) == 1:
#         pred_text = _render_predicate_detailed(predicates[0])
#         return f"filters rows where {pred_text}"
    
#     pred_texts = [_render_predicate_detailed(p) for p in predicates]
    
#     if len(pred_texts) == 2:
#         return f"filters rows where {pred_texts[0]}, AND {pred_texts[1]}"
    
#     # Multiple conditions
#     formatted = []
#     for i, pt in enumerate(pred_texts):
#         if i == 0:
#             formatted.append(f"filters rows where {pt}")
#         elif i < len(pred_texts) - 1:
#             formatted.append(f", AND {pt}")
#         else:
#             formatted.append(f", AND finally {pt}")
#     return "".join(formatted)


# def _explain_join_conditions(raw_condition, join_type=None):
#     """Generate detailed explanation of join conditions."""
#     # Explain join type first
#     join_phrase = ""
#     if join_type:
#         jt = str(join_type).strip().upper().replace("_", " ")
#         type_map = {
#             "INNER": "an INNER join (keeping only matching rows from both sides)",
#             "LEFT": "a LEFT OUTER join (keeping all rows from the left table, with NULLs for non-matches)",
#             "RIGHT": "a RIGHT OUTER join (keeping all rows from the right table, with NULLs for non-matches)",
#             "FULL": "a FULL OUTER join (keeping all rows from both tables, with NULLs for non-matches)",
#             "CROSS": "a CROSS join (Cartesian product)",
#         }
#         join_phrase = type_map.get(jt, f"a {jt} join")
#     else:
#         join_phrase = "a join"
    
#     if not raw_condition:
#         return f"performs {join_phrase} (without explicit conditions)"
    
#     conjuncts = split_conjuncts(raw_condition) if isinstance(raw_condition, str) else [raw_condition.get("expr", "")]
#     if not conjuncts:
#         return f"performs {join_phrase}"
    
#     predicates = [parse_predicate(c) for c in conjuncts]
    
#     if len(predicates) == 1:
#         pred_text = _render_predicate_detailed(predicates[0])
#         return f"performs {join_phrase}, matching rows where {pred_text}"
    
#     pred_texts = [_render_predicate_detailed(p) for p in predicates]
    
#     if len(pred_texts) == 2:
#         return f"performs {join_phrase}, matching where {pred_texts[0]} AND {pred_texts[1]}"
    
#     all_but_last = ", ".join(pred_texts[:-1])
#     return f"performs {join_phrase}, matching where {all_but_last}, AND {pred_texts[-1]}"


# def _explain_projection(expressions):
#     """Generate detailed explanation of projection expressions."""
#     exprs = _ensure_list_and_get_exprs(expressions)
    
#     if not exprs:
#         return "selects specific columns"
    
#     # Clean and explain expressions
#     explained = [_explain_expression(e) for e in exprs[:5]]  # Show first 5
    
#     if len(explained) == 1:
#         return f"selects: {explained[0]}"
    
#     if len(exprs) <= 5:
#         all_but_last = ", ".join(explained[:-1])
#         return f"selects: {all_but_last}, and {explained[-1]}"
    
#     # Many columns
#     first_few = ", ".join(explained[:3])
#     return f"selects {len(exprs)} columns including: {first_few}, and {len(exprs)-3} others"


# def _explain_aggregate(expressions, groups):
#     """Generate detailed explanation of aggregation operations."""
#     agg_exprs = _ensure_list_and_get_exprs(expressions)
#     group_exprs = _ensure_list_and_get_exprs(groups)
    
#     # Handle DISTINCT-only aggregates
#     if not agg_exprs or (len(agg_exprs) == 1 and str(agg_exprs[0]).strip().upper() == "DISTINCT"):
#         if group_exprs:
#             groups_text = ", ".join([_explain_expression(g) for g in group_exprs[:3]])
#             return f"computes distinct combinations of: {groups_text}"
#         return "computes distinct rows"
    
#     # Parse aggregation expressions
#     agg_parts = []
#     for expr in agg_exprs[:5]:  # Limit to first 5
#         explained = _explain_expression(str(expr))
#         expr_lower = str(expr).lower()
        
#         if "count(*)" in expr_lower or "count( * )" in expr_lower:
#             agg_parts.append("counts all rows")
#         elif "count" in expr_lower:
#             agg_parts.append(f"counts {explained}")
#         elif "sum" in expr_lower:
#             agg_parts.append(f"sums {explained}")
#         elif "avg" in expr_lower:
#             agg_parts.append(f"computes average of {explained}")
#         elif "min" in expr_lower:
#             agg_parts.append(f"finds minimum {explained}")
#         elif "max" in expr_lower:
#             agg_parts.append(f"finds maximum {explained}")
#         else:
#             agg_parts.append(f"computes {explained}")
    
#     if not group_exprs:
#         if len(agg_parts) == 1:
#             return f"aggregates data: {agg_parts[0]} across all rows"
#         all_but_last = ", ".join(agg_parts[:-1])
#         return f"aggregates data: {all_but_last}, and {agg_parts[-1]}"
    
#     # With grouping
#     group_clean = [_explain_expression(g) for g in group_exprs[:3]]
#     if len(group_clean) == 1:
#         group_text = f"grouped by {group_clean[0]}"
#     else:
#         all_but_last = ", ".join(group_clean[:-1])
#         group_text = f"grouped by {all_but_last} and {group_clean[-1]}"
    
#     if len(agg_parts) == 1:
#         return f"aggregates data ({group_text}): {agg_parts[0]}"
    
#     all_but_last = ", ".join(agg_parts[:-1])
#     return f"aggregates data ({group_text}): {all_but_last}, and {agg_parts[-1]}"


# def _explain_sort(order_exprs, limit=None, offset=None):
#     """Generate detailed explanation of sorting operations."""
#     orders = _ensure_list_and_get_exprs(order_exprs)
    
#     if not orders:
#         order_text = "sorts the results"
#     else:
#         order_parts = []
#         for o in orders[:3]:  # Limit to first 3
#             o_str = str(o).strip()
            
#             # Extract direction
#             desc_match = re.search(r"\s+(desc|asc)$", o_str, re.IGNORECASE)
#             if desc_match:
#                 direction = desc_match.group(1).lower()
#                 col = o_str[:desc_match.start()].strip()
#             else:
#                 direction = "asc"
#                 col = o_str
            
#             col_explained = _explain_expression(col)
#             dir_text = "descending" if direction == "desc" else "ascending"
#             order_parts.append(f"{col_explained} in {dir_text} order")
        
#         if len(order_parts) == 1:
#             order_text = f"sorts results by {order_parts[0]}"
#         elif len(orders) <= 3:
#             all_but_last = ", then by ".join(order_parts[:-1])
#             order_text = f"sorts results by {all_but_last}, then by {order_parts[-1]}"
#         else:
#             order_text = f"sorts results by {order_parts[0]}, then by {order_parts[1]}, and {len(orders)-2} more criteria"
    
#     # Add offset and limit
#     suffix = []
#     if offset and str(offset) != "0":
#         suffix.append(f"skipping the first {offset} rows")
#     if limit:
#         suffix.append(f"keeping only the top {limit} rows")
    
#     if suffix:
#         return f"{order_text}, {' and '.join(suffix)}"
#     return order_text


# def _get_base_tables(node):
#     """Extract all base table names from scan nodes in the subtree."""
#     tables = []
    
#     def walk(n):
#         if not n:
#             return
#         name = (n.get("name") or "").upper()
#         if "SCAN" in name:
#             table = (n.get("extra_info") or {}).get("Table")
#             if table and table not in tables:
#                 tables.append(table)
#         for child in n.get("children", []):
#             walk(child)
    
#     walk(node)
#     return tables


# def serialize_enhanced_preorder_cot(resolved_ra_tree):
#     """
#     Enhanced pre-order serialization generating detailed Chain-of-Thought explanations.
#     Handles all corner cases including quoted identifiers, CAST, AND(), subqueries, etc.
    
#     Args:
#         resolved_ra_tree: RA tree with pointers already resolved
        
#     Returns:
#         str: Detailed natural language explanation
#     """
    
#     # Collect all base tables
#     base_tables = _get_base_tables(resolved_ra_tree)
    
#     lines = []
#     lines.append("**Query Plan Explanation:**\n")
    
#     # Add table context
#     if base_tables:
#         if len(base_tables) == 1:
#             lines.append(f"This query works with data from the **{base_tables[0]}** table.\n")
#         else:
#             table_list = "**, **".join(base_tables[:-1]) + f"**, and **{base_tables[-1]}**"
#             lines.append(f"This query works with data from these tables: **{table_list}**.\n")
    
#     step_num = 1
    
#     def traverse_and_explain(node, is_root=True):
#         nonlocal step_num
        
#         name = (node.get("name") or "").upper()
#         info = node.get("extra_info", {}) or {}
#         children = node.get("children", [])
        
#         # Skip leaf scan nodes (already mentioned in context)
#         if "SCAN" in name:
#             for child in children:
#                 traverse_and_explain(child, False)
#             return
        
#         # Skip VALUES nodes
#         if name == "VALUES":
#             return
        
#         # Build explanation based on operator type
#         explanation = ""
        
#         if name == "PROJECTION":
#             explanation = _explain_projection(info.get("Expressions"))
        
#         elif name == "FILTER":
#             explanation = _explain_filter_conditions(info.get("Filters"))
        
#         elif name in ("JOIN", "COMPARISON_JOIN", "ANY_JOIN", "DEPENDENT_JOIN"):
#             explanation = _explain_join_conditions(
#                 info.get("Condition"),
#                 info.get("JoinType")
#             )
        
#         elif name == "CROSS_PRODUCT":
#             left_tables = _get_base_tables(children[0]) if children else []
#             right_tables = _get_base_tables(children[1]) if len(children) > 1 else []
#             lt = left_tables[0] if left_tables else "the left source"
#             rt = right_tables[0] if right_tables else "the right source"
#             explanation = f"creates a Cartesian product between **{lt}** and **{rt}**, pairing every row from one with every row from the other"
        
#         elif name == "AGGREGATE":
#             explanation = _explain_aggregate(
#                 info.get("Expressions"),
#                 info.get("Groups")
#             )
        
#         elif name in ("SORT", "ORDER_BY"):
#             explanation = _explain_sort(
#                 info.get("Order"),
#                 info.get("Limit"),
#                 info.get("Offset")
#             )
        
#         elif name == "LIMIT":
#             limit_val = info.get("Limit", "N")
#             explanation = f"limits the result to return only the top {limit_val} rows"
        
#         elif name == "DISTINCT":
#             explanation = "removes duplicate rows, ensuring each unique combination appears only once"
        
#         elif name == "UNION":
#             explanation = "combines rows from two result sets using UNION (keeping unique rows from both)"
        
#         elif name == "INTERSECT":
#             explanation = "finds the intersection of two result sets (rows that appear in both)"
        
#         elif name == "EXCEPT":
#             explanation = "computes the set difference (rows in the first set that are NOT in the second)"
        
#         else:
#             explanation = f"applies the **{name}** operation"
        
#         # Add step to output
#         if is_root:
#             lines.append(f"**Step {step_num} (Root):** {explanation.capitalize()}.\n")
#         else:
#             lines.append(f"**Step {step_num}:** {explanation.capitalize()}.\n")
        
#         step_num += 1
        
#         # Recursively process children
#         for child in children:
#             traverse_and_explain(child, False)
    
#     # Start traversal from root
#     traverse_and_explain(resolved_ra_tree, is_root=True)
    
#     # Add conclusion
#     lines.append("\n**Result:** The query returns the output from the root operation.")
    
#     return "".join(lines)


# # Compact version
# def serialize_compact_preorder_cot(resolved_ra_tree):
#     """Compact version with numbered steps."""
#     base_tables = _get_base_tables(resolved_ra_tree)
    
#     lines = ["**Query Plan:**\n"]
    
#     if base_tables:
#         tables = "', '".join(base_tables)
#         lines.append(f"Tables: '{tables}'\n\n")
    
#     step_num = 1
    
#     def traverse(node):
#         nonlocal step_num
        
#         name = (node.get("name") or "").upper()
#         info = node.get("extra_info", {}) or {}
        
#         if "SCAN" in name or name == "VALUES":
#             for child in node.get("children", []):
#                 traverse(child)
#             return
        
#         # Generate explanation
#         if name == "PROJECTION":
#             exp = _explain_projection(info.get("Expressions"))
#         elif name == "FILTER":
#             exp = _explain_filter_conditions(info.get("Filters"))
#         elif name in ("JOIN", "COMPARISON_JOIN", "ANY_JOIN", "DEPENDENT_JOIN"):
#             exp = _explain_join_conditions(info.get("Condition"), info.get("JoinType"))
#         elif name == "AGGREGATE":
#             exp = _explain_aggregate(info.get("Expressions"), info.get("Groups"))
#         elif name in ("SORT", "ORDER_BY"):
#             exp = _explain_sort(info.get("Order"), info.get("Limit"), info.get("Offset"))
#         elif name == "DISTINCT":
#             exp = "removes duplicates"
#         else:
#             exp = f"applies {name}"
        
#         lines.append(f"{step_num}. {exp.capitalize()}.\n")
#         step_num += 1
        
#         for child in node.get("children", []):
#             traverse(child)
    
#     traverse(resolved_ra_tree)
#     return "".join(lines)
