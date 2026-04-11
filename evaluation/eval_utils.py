import json
import re
from nltk import word_tokenize
import nltk

class RATreeNode:
    """A node in the Relational Algebra Tree."""
    def __init__(self, name, extra_info=None, children=None, path=""):
        self.name = name
        self.extra_info = extra_info or {}
        self.children = children or []
        self.path = path  # Path to this node from the root

    def __str__(self):
        """Returns a canonical string representation of the node."""
        extra_info_str = ",".join([f"{k}:{self.extra_info[k]}" for k in sorted(self.extra_info.keys())])
        child_str = ",".join(sorted([str(c) for c in self.children]))
        return f"{self.name}({extra_info_str})[{child_str}]"

    def to_dict(self):
        """Converts the node to a dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "extra_info": self.extra_info,
            "children": [c.to_dict() for c in self.children],
        }

def parse_ra(ra_json, path="ROOT"):
    """
    Parses a relational algebra JSON object into a canonical tree structure.
    """
    if not ra_json:
        return None
    name = ra_json.get("name")
    extra_info = ra_json.get("extra_info", {})
    children = ra_json.get("children", [])
    
    for key, value in extra_info.items():
        if isinstance(value, list):
            try:
                extra_info[key] = sorted(map(str, value))
            except (TypeError, ValueError):
                pass
    
    # Assign paths to children before sorting
    parsed_children = [parse_ra(child, f"{path}/{name}[{i}]") for i, child in enumerate(children)]

    commutative_ops = {"JOIN", "CROSS_PRODUCT", "INTERSECT", "UNION", "COMPARISON_JOIN"}
    if name in commutative_ops:
        parsed_children.sort(key=lambda x: str(x))

    return RATreeNode(name, extra_info, parsed_children, path)

def tokenize_ra_expression(expr_str):
    """
    Intelligently tokenizes a relational algebra expression.
    - Deep parse for complex expressions using NLTK.
    - Atomic treatment for simple identifiers or symbolic expressions.
    """
    expr_str = str(expr_str)
    # Heuristic: If the expression looks complex (has operators/parentheses), use NLTK.
    if re.search(r'[<>=!()]', expr_str):
        expr_str = expr_str.replace("`", "'").replace("\"", "'")
        quote_idxs = [idx for idx, char in enumerate(expr_str) if char == "'"]
        assert len(quote_idxs) % 2 == 0, f"Mismatched quotes in expression: {expr_str}"

        vals = {}
        for i in range(len(quote_idxs) - 1, -1, -2):
            qidx1, qidx2 = quote_idxs[i-1], quote_idxs[i]
            val = expr_str[qidx1:qidx2+1]; key = f"__VAL_{i}__"
            expr_str = expr_str[:qidx1] + key + expr_str[qidx2+1:]; vals[key] = val
        
        toks = [word.lower() for word in nltk.word_tokenize(expr_str)]

        for i in range(len(toks)):
            if toks[i] in vals: toks[i] = vals[toks[i]]

        eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
        for eq_idx in reversed(eq_idxs):
            if eq_idx > 0 and toks[eq_idx - 1] in ['!', '>', '<']:
                toks[eq_idx - 1] += "="; del toks[eq_idx]
        return tuple(toks)
    else:
        # Otherwise, treat the entire string as a single, atomic token.
        return (expr_str.lower(),)


# def tokenize_ra_expression(expr_str):
#     """
#     Tokenizes a relational algebra expression using NLTK, properly handling string literals.
#     This logic is adapted from the official Spider evaluation script.
#     """
#     expr_str = str(expr_str)
#     # Replace different quote types with a standard one
#     expr_str = expr_str.replace("`", "'").replace("\"", "'")
    
#     # Isolate string literals to treat them as single tokens
#     quote_idxs = [idx for idx, char in enumerate(expr_str) if char == "'"]
#     assert len(quote_idxs) % 2 == 0, "Mismatched quotes in expression."

#     # Store literals and replace them with placeholders
#     vals = {}
#     for i in range(len(quote_idxs) - 1, -1, -2):
#         qidx1, qidx2 = quote_idxs[i-1], quote_idxs[i]
#         val = expr_str[qidx1:qidx2+1]
#         key = f"__VAL_{i}__"
#         expr_str = expr_str[:qidx1] + key + expr_str[qidx2+1:]
#         vals[key] = val
    
#     # Tokenize the modified string
#     toks = [word.lower() for word in word_tokenize(expr_str)]

#     # Restore the string literals
#     for i in range(len(toks)):
#         if toks[i] in vals:
#             toks[i] = vals[toks[i]]

#     # Combine multi-part operators (e.g., '>', '=')
#     eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
#     for eq_idx in reversed(eq_idxs):
#         if eq_idx > 0 and toks[eq_idx - 1] in ['!', '>', '<']:
#             toks[eq_idx - 1] = toks[eq_idx - 1] + "="
#             del toks[eq_idx]
            
#     return tuple(toks) # Return as a tuple to make it hashable for sets

def exact_match(predicted_ra, ground_truth_ra):
    """Performs an exact match comparison between two RA trees."""
    return str(predicted_ra) == str(ground_truth_ra)


def score_node(pred_node, gt_node):
    """Scores a single node based on its type and extra_info."""
    if pred_node.name != gt_node.name:
        return 0.0, [{"component": "OPERATOR_MISMATCH", "score": 0.0, "location": gt_node.path}]

    score_func_map = {
        "PROJECTION": score_projection,
        "FILTER": score_filter,
        "JOIN": score_join,
        "AGGREGATE": score_aggregate,
        "SORT": score_sort,
        "ORDER_BY": score_sort,
        "GROUP_BY": score_group_by,
        "LIMIT": score_limit,
        "DISTINCT": score_distinct,
    }
    score_func = score_func_map.get(pred_node.name, score_default)
    return score_func(pred_node, gt_node)



def jaccard_similarity_on_token_tuples(set1, set2):
    if not set1 and not set2: return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 0.0

def score_projection(pred_node, gt_node):
    pred_toks = {tokenize_ra_expression(e) for e in pred_node.extra_info.get("Expressions", [])}
    gt_toks = {tokenize_ra_expression(e) for e in gt_node.extra_info.get("Expressions", [])}
    print("Projection - Pred:", pred_toks)
    print("Projection - GT:", gt_toks)
    score = jaccard_similarity_on_token_tuples(pred_toks, gt_toks)
    return score, [{"component": "PROJECTION_expressions", "score": score, "location": gt_node.path}]

def score_filter(pred_node, gt_node):
    pred_toks = {tokenize_ra_expression(c) for c in pred_node.extra_info.get("Expressions", [])}
    gt_toks = {tokenize_ra_expression(c) for c in gt_node.extra_info.get("Expressions", [])}
    score = jaccard_similarity_on_token_tuples(pred_toks, gt_toks)
    return score, [{"component": "FILTER_conditions", "score": score, "location": gt_node.path}]

def score_join(pred_node, gt_node):
    pred_conds = pred_node.extra_info.get("Conditions", "").split(" AND ")
    gt_conds = gt_node.extra_info.get("Conditions", "").split(" AND ")
    pred_toks = {tokenize_ra_expression(c) for c in pred_conds if c}
    gt_toks = {tokenize_ra_expression(c) for c in gt_conds if c}
    score = jaccard_similarity_on_token_tuples(pred_toks, gt_toks)
    return score, [{"component": "JOIN_conditions", "score": score, "location": gt_node.path}]
    
def score_aggregate(pred_node, gt_node):
    pred_expr_toks = {tokenize_ra_expression(e) for e in pred_node.extra_info.get("Expressions", [])}
    gt_expr_toks = {tokenize_ra_expression(e) for e in gt_node.extra_info.get("Expressions", [])}
    expr_score = jaccard_similarity_on_token_tuples(pred_expr_toks, gt_expr_toks)

    pred_group_toks = {tokenize_ra_expression(g) for g in pred_node.extra_info.get("Grouping Expressions", [])}
    gt_group_toks = {tokenize_ra_expression(g) for g in gt_node.extra_info.get("Grouping Expressions", [])}
    group_score = jaccard_similarity_on_token_tuples(pred_group_toks, gt_group_toks)
    
    score = 0.7 * expr_score + 0.3 * group_score
    return score, [
        {"component": "AGGREGATE_expressions", "score": expr_score, "location": gt_node.path},
        {"component": "AGGREGATE_grouping", "score": group_score, "location": gt_node.path}
    ]

def score_sort(pred_node, gt_node):
    pred_toks = {tokenize_ra_expression(o) for o in pred_node.extra_info.get("Order By", [])}
    gt_toks = {tokenize_ra_expression(o) for o in gt_node.extra_info.get("Order By", [])}
    score = jaccard_similarity_on_token_tuples(pred_toks, gt_toks)
    return score, [{"component": "SORT_orderby", "score": score, "location": gt_node.path}]

def score_group_by(pred_node, gt_node):
    pred_toks = {tokenize_ra_expression(g) for g in pred_node.extra_info.get("Grouping Expressions", [])}
    gt_toks = {tokenize_ra_expression(g) for g in gt_node.extra_info.get("Grouping Expressions", [])}
    score = jaccard_similarity_on_token_tuples(pred_toks, gt_toks)
    return score, [{"component": "GROUPBY_expressions", "score": score, "location": gt_node.path}]


def score_limit(pred_node, gt_node):
    """Scores a LIMIT node with a graded approach."""
    pred_limit_str = pred_node.extra_info.get("Expressions")
    gt_limit_str = gt_node.extra_info.get("Expressions")
    
    try:
        pred_limit = int(pred_limit_str)
        gt_limit = int(gt_limit_str)
        
        if gt_limit == 0:
            return 1.0 if pred_limit == 0 else 0.0, [{"component": "LIMIT_value", "score": 1.0 if pred_limit == 0 else 0.0, "location": gt_node.path}]

        # Normalized distance score
        score = max(0, 1 - abs(pred_limit - gt_limit) / gt_limit)
    except (ValueError, TypeError):
        score = 1.0 if pred_limit_str == gt_limit_str else 0.0
        
    return score, [{"component": "LIMIT_value", "score": score, "location": gt_node.path}]

def score_distinct(pred_node, gt_node):
    return 1.0, [{"component": "DISTINCT_operator", "score": 1.0, "location": gt_node.path}]

def score_default(pred_node, gt_node):
    if pred_node.extra_info.get("Table") == gt_node.extra_info.get("Table"):
        return 1.0, [{"component": "TABLE_match", "score": 1.0, "location": gt_node.path}]
    return 0.0, [{"component": "TABLE_match", "score": 0.0, "location": gt_node.path}]

def count_gt_components(gt_node):
    """Recursively counts the number of scorable components in the ground truth tree."""
    count = 0
    if gt_node.name == "AGGREGATE":
        count += 2  # Expressions and grouping
    else:
        count += 1
    
    for child in gt_node.children:
        count += count_gt_components(child)
    return count

def evaluate(predicted_ra, ground_truth_ra):
    """Evaluates the predicted RA tree against the ground truth with partial scoring."""
    if not predicted_ra and not ground_truth_ra:
        return {"score": 1.0, "components": [], "component_recall_score": 1.0}
    if not predicted_ra or not ground_truth_ra:
        total_gt_components = count_gt_components(ground_truth_ra) if ground_truth_ra else 0
        return {"score": 0.0, "components": [], "component_recall_score": 0.0}

    # First, get the detailed component scores
    eval_result = _evaluate_recursive(predicted_ra, ground_truth_ra)

    # Now, calculate the component recall score
    total_gt_components = count_gt_components(ground_truth_ra)
    # print(f"Total ground truth components: {total_gt_components}")
    if total_gt_components == 0:
        component_recall_score = 1.0
    else:
        sum_of_component_scores = sum(c['score'] for c in eval_result['components'] if c['component'] != "child_structure_mismatch")
        # print(f"Sum of component scores: {sum_of_component_scores}")
        component_recall_score = sum_of_component_scores / total_gt_components

    eval_result["component_recall_score"] = component_recall_score
    return eval_result

def _evaluate_recursive(predicted_ra, ground_truth_ra):
    """Recursive helper for evaluation."""
    if not predicted_ra and not ground_truth_ra:
        return {"score": 1.0, "components": []}
    if not predicted_ra or not ground_truth_ra:
        return {"score": 0.0, "components": []}

    node_score, component_scores = score_node(predicted_ra, ground_truth_ra)

    if len(predicted_ra.children) != len(ground_truth_ra.children):
        avg_child_score = 0.0
        component_scores.append({"component": "child_structure_mismatch", "score": 0.0, "location": ground_truth_ra.path})
    else:
        child_scores = []
        if predicted_ra.children:
            for pred_child, gt_child in zip(predicted_ra.children, ground_truth_ra.children):
                child_eval = _evaluate_recursive(pred_child, gt_child)
                child_scores.append(child_eval["score"])
                component_scores.extend(child_eval["components"])
            avg_child_score = sum(child_scores) / len(child_scores)
            component_scores.append({"component": "child_structure_mismatch", "score": 1.0, "location": ground_truth_ra.path})
        else:
            avg_child_score = 1.0


    final_score = 0.5 * node_score + 0.5 * avg_child_score
    return {"score": final_score, "components": component_scores}

def ra_eval_res(ground_truth_ra_json, predicted_ra_json):

    ground_truth_tree = parse_ra(ground_truth_ra_json)
    predicted_tree = parse_ra(predicted_ra_json)

    evaluation_result = evaluate(predicted_tree, ground_truth_tree)

    return evaluation_result

def print_eval_res(ground_truth_ra_json, predicted_ra_json):

    ground_truth_tree = parse_ra(ground_truth_ra_json)
    predicted_tree = parse_ra(predicted_ra_json)

    evaluation_result = evaluate(predicted_tree, ground_truth_tree)
    print("\n--- Relational Algebra Evaluation Report ---")
    print(f"\n  Overall Tree Score: {evaluation_result['score']:.2f}")
    print(f"  Component Recall Score: {evaluation_result['component_recall_score']:.2f}")

    print("\n  Component Score Breakdown:")
    for item in sorted(evaluation_result['components'], key=lambda x: x['location']):
        print(f"    - Location: {item['location']}")
        print(f"      Component: {item['component']}")
        print(f"      Score: {item['score']:.2f}")

    return evaluation_result


def normalize_sql_query(sql: str) -> str:
    """Normalize SQL query for comparison.

    - Remove extra whitespace
    - Convert to uppercase
    - Remove trailing semicolon
    - Normalize spacing around operators
    """
    # Remove comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

    # Convert to uppercase
    sql = sql.upper()

    # Remove trailing semicolon
    sql = sql.rstrip(';').strip()

    # Normalize whitespace
    sql = ' '.join(sql.split())

    # Normalize spacing around common operators
    sql = re.sub(r'\s*([=<>!]+)\s*', r' \1 ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\s*\(\s*', ' (', sql)
    sql = re.sub(r'\s*\)\s*', ') ', sql)

    # Clean up multiple spaces
    sql = ' '.join(sql.split())

    return sql.strip()


def parse_ra_output(text: str, extract_answer_func=None, extract_json_func=None):
    """Parse relational algebra JSON from model output.

    Uses the centralized extraction functions for consistency.
    First tries to extract from answer section, then from full text.

    Args:
        text: Raw model output text
        extract_answer_func: Function to extract answer content (optional)
        extract_json_func: Function to extract JSON from text (required)
    """
    if not extract_json_func:
        raise ValueError("extract_json_func is required")

    try:
        # First try to extract from answer section if present
        if extract_answer_func:
            answer_content = extract_answer_func(text)
            if answer_content:
                result = extract_json_func(answer_content)
                if result:
                    return result

        # If not found in answer section, try whole text
        return extract_json_func(text)

    except Exception:
        return None


def parse_sql_output(text: str, extract_answer_func=None, extract_sql_func=None):
    """Parse SQL query from model output.

    Uses the centralized extraction functions for consistency.
    First tries to extract from answer section, then from full text.

    Args:
        text: Raw model output text
        extract_answer_func: Function to extract answer content (optional)
        extract_sql_func: Function to extract SQL from text (required)
    """
    if not extract_sql_func:
        raise ValueError("extract_sql_func is required")

    try:
        # First try to extract from answer section if present
        if extract_answer_func:
            answer_content = extract_answer_func(text)
            if answer_content:
                result = extract_sql_func(answer_content)
                if result:
                    return result

        # If not found in answer section, try whole text
        return extract_sql_func(text)

    except Exception:
        return None