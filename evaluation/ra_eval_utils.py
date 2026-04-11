import re
import json
import ast

# =========================
# Node + Parser (unchanged)
# =========================

class RATreeNode:
    """A node in the Relational Algebra Tree."""
    def __init__(self, name, extra_info=None, children=None, path=""):
        self.name = name
        self.extra_info = extra_info or {}
        self.children = children or []
        self.path = path

    def __str__(self):
        extra_info_str = ",".join([f"{k}:{self.extra_info[k]}" for k in sorted(self.extra_info.keys())])
        child_str = ",".join(sorted([str(c) for c in self.children]))
        return f"{self.name}({extra_info_str})[{child_str}]"

    def to_dict(self):
        return {
            "name": self.name,
            "path": self.path,
            "extra_info": self.extra_info,
            "children": [c.to_dict() for c in self.children],
        }

def parse_ra(ra_json, path="ROOT"):
    if not ra_json:
        return None
    name = ra_json.get("name")
    extra_info = ra_json.get("extra_info", {})
    children = ra_json.get("children", [])

    for key, value in list(extra_info.items()):
        if isinstance(value, list):
            try:
                extra_info[key] = sorted(map(str, value))
            except (TypeError, ValueError):
                pass

    parsed_children = [parse_ra(child, f"{path}/{name}[{i}]") for i, child in enumerate(children)]
    commutative_ops = {"JOIN", "CROSS_PRODUCT", "INTERSECT", "UNION", "COMPARISON_JOIN"}
    if name in commutative_ops:
        parsed_children.sort(key=lambda x: str(x))
    return RATreeNode(name, extra_info, parsed_children, path)

# ===========================================
# Letter-level tokenizer & join 'AND' splitter
# ===========================================

IGNORE_LITERAL_VALUES = True
MASK_STR = "__str__"
MASK_NUM = "__num__"
_MULTIWORD_OPS = [("is", "not", "null", "is_not_null"),("is", "null", "is_null"),("not", "in", "not_in"),("order", "by", "order_by"),("group", "by", "group_by"),("left", "outer", "left_outer"),("right", "outer", "right_outer"),("full", "outer", "full_outer")]
_TWOCHAR_OPS = {"<=", ">=", "!=", "<>", "=="}
_ONECHAR_OPS = {"=", "<", ">", "+", "-", "*", "/", "%"}
_PARENS = {"(", ")"}
_PUNCT = {",", "."}
# Reserved tokens that should not be treated as identifiers
RESERVED_TOKENS = {
    "and", "or", "not", "in", "between", "like", "is",
    "is_null", "is_not_null",
    "order_by", "group_by", "asc", "desc",
    "null", "true", "false",
    "cast", "case", "when", "then", "else", "end",
    "as", "on", "join", "inner", "left", "right", "full", "outer",
    "distinct"
}
_ID_START = re.compile(r"[A-Za-z_]")
_ID_CONT  = re.compile(r"[A-Za-z0-9_\$]")
_DIGIT    = re.compile(r"[0-9]")
_WS       = re.compile(r"\s+")

def _read_string(s, i):
    q = s[i]; i += 1; buf = [q]; escaped = False
    while i < len(s):
        ch = s[i]; buf.append(ch); i += 1
        if escaped: escaped = False; continue
        if ch == "\\": escaped = True; continue
        if ch == q: break
    return "".join(buf), i

def _read_number(s, i):
    start = i
    while i < len(s) and _DIGIT.match(s[i]): i += 1
    if i < len(s) and s[i] == ".":
        i += 1
        while i < len(s) and _DIGIT.match(s[i]): i += 1
    if i < len(s) and s[i] in "eE":
        j = i + 1
        if j < len(s) and s[j] in "+-": j += 1
        if j < len(s) and _DIGIT.match(s[j]) is not None:
            i = j + 1
            while i < len(s) and _DIGIT.match(s[i]): i += 1
    return s[start:i], i

def _read_identifier(s, i):
    start = i; i += 1
    while i < len(s) and _ID_CONT.match(s[i]): i += 1
    return s[start:i], i

def _lex_ra(s):
    toks = []; i = 0; n = len(s)
    while i < n:
        m = _WS.match(s, i)
        if m: i = m.end(); continue
        ch = s[i]
        if ch in ("'", '"', "`"): lit, i = _read_string(s, i); toks.append(lit); continue
        if i + 1 < n and s[i:i+2] in _TWOCHAR_OPS: toks.append(s[i:i+2]); i += 2; continue
        if ch in _ONECHAR_OPS or ch in _PARENS or ch in _PUNCT: toks.append(ch); i += 1; continue
        if _DIGIT.match(ch): lit, i = _read_number(s, i); toks.append(lit); continue
        if _ID_START.match(ch): ident, i = _read_identifier(s, i); toks.append(ident); continue
        toks.append(ch); i += 1
    return toks

def _merge_dotted_identifiers(toks):
    out = []; i = 0
    while i < len(toks):
        if i + 2 < len(toks) and toks[i+1] == ".":
            out.append(f"{toks[i]}.{toks[i+2]}"); i += 3
        else:
            out.append(toks[i]); i += 1
    return out

def _lower_and_mask(toks):
    lowered = []
    for t in toks:
        if len(t) >= 2 and ((t[0] in ("'", '"', "`") and t[-1] == t[0])):
            lowered.append(MASK_STR if IGNORE_LITERAL_VALUES else t.lower())
        else:
            try:
                float(t)
                lowered.append(MASK_NUM if IGNORE_LITERAL_VALUES else t.lower())
            except ValueError:
                lowered.append(t.lower())
    return lowered

def _combine_multiword_ops(toks):
    out = []; i = 0; n = len(toks)
    while i < n:
        matched = False
        for pat in _MULTIWORD_OPS:
            L, repl, seq = len(pat) - 1, pat[-1], pat[:-1]
            if i + L <= n and tuple(toks[i:i+L]) == seq:
                out.append(repl); i += L; matched = True; break
        if not matched: out.append(toks[i]); i += 1
    return out

def _split_top_level(tokens, ops):
    depth = 0
    for i, t in enumerate(tokens):
        if t == "(": depth += 1
        elif t == ")": depth = max(depth - 1, 0)
        elif depth == 0 and t in ops: return tokens[:i], t, tokens[i+1:]
    return None

def _balanced_parens(ts):
    d = 0
    for t in ts:
        if t == "(": d += 1
        elif t == ")": d -= 1
        if d < 0: return False
    return d == 0

def _peel_outer_parens(toks):
    if len(toks) >= 2 and toks[0] == "(" and toks[-1] == ")" and _balanced_parens(toks):
        return toks[1:-1]
    return toks

def _canonicalize_equals(tokens):
    # Only canonicalize equality when BOTH sides are identifier-ish (e.g., join keys)
    split = _split_top_level(tokens, {"=", "=="})
    if not split:
        return tokens
    left, op, right = split

    # If either side contains a literal/number mask, do NOT reorder
    if "__str__" in left or "__num__" in left or "__str__" in right or "__num__" in right:
        return tokens

    # Only reorder when both sides contain at least one identifier
    def has_ident(ts):
        return any(_is_ident(t) for t in ts)

    if not (has_ident(left) and has_ident(right)):
        return tokens

    L, R = " ".join(left), " ".join(right)
    if R < L:
        left, right = right, left
    return left + [op] + right

def tokenize_ra_expression(expr_str, mask_literals=True, canonicalize=True):
    if expr_str is None: return tuple()
    toks = _lex_ra(str(expr_str))
    toks = _merge_dotted_identifiers(toks)
    # Respect mask_literals flag
    if mask_literals:
        toks = _lower_and_mask(toks)
    else:
        # Lowercase without masking
        toks = [t.lower() for t in toks]
    toks = _combine_multiword_ops(toks)
    if canonicalize:
        toks = _canonicalize_equals(toks)
    toks = _peel_outer_parens(toks)
    return tuple(toks)

def split_conjuncts(expr):
    if not expr: return []
    toks = _lex_ra(str(expr))

    # Handle functional form: AND(expr1, expr2, ...)
    if len(toks) >= 3 and isinstance(toks[0], str) and toks[0].lower() == "and" and toks[1] == "(" and toks[-1] == ")" and _balanced_parens(toks[1:]):
        inner = toks[2:-1]
        parts = []
        buf = []
        depth = 0
        for t in inner:
            if t == "(":
                depth += 1
                buf.append(t)
            elif t == ")":
                depth = max(0, depth - 1)
                buf.append(t)
            elif t == "," and depth == 0:
                part = " ".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(t)
        if buf:
            part = " ".join(buf).strip()
            if part:
                parts.append(part)
        return parts

    # Fallback: infix AND at top-level
    parts = []; buf = []; depth = 0
    for t in toks:
        if t == "(": depth += 1
        elif t == ")": depth = max(depth - 1, 0)
        elif depth == 0 and isinstance(t, str) and t.lower() == "and":
            parts.append(" ".join(buf).strip()); buf = []
            continue
        buf.append(t)
    if buf: parts.append(" ".join(buf).strip())
    return [p for p in parts if p]

# =======================
# Scoring & evaluation
# =======================

def exact_match(predicted_ra, ground_truth_ra):
    return str(predicted_ra) == str(ground_truth_ra)

def _ensure_list_and_get_exprs(v):
    if v is None:
        return []
    # Normalize input to list
    v = v if isinstance(v, list) else [v]
    expr_list = []
    for item in v:
        # Case 1: dict with 'expr'
        if isinstance(item, dict) and "expr" in item:
            expr_list.append(item["expr"])
            continue
        # Case 2: string that may be a stringified dict/list; try to parse
        if isinstance(item, str):
            s = item.strip()
            if s and s[0] in "{[":
                try:
                    parsed = ast.literal_eval(s)
                    # dict with 'expr'
                    if isinstance(parsed, dict) and "expr" in parsed:
                        expr_list.append(parsed["expr"])
                        continue
                    # list of dicts/strings
                    if isinstance(parsed, list):
                        for it in parsed:
                            if isinstance(it, dict) and "expr" in it:
                                expr_list.append(it["expr"])
                            elif isinstance(it, str):
                                expr_list.append(it)
                        continue
                except Exception:
                    # Fall through to treat as raw string
                    pass
            # treat plain string as an expression
            expr_list.append(item)
    return expr_list

def jaccard_similarity_on_token_tuples(set1, set2):
    if not set1 and not set2: return 1.0
    U = set1.union(set2)
    return (len(set1.intersection(set2)) / len(U)) if U else 0.0

_AGG_FUNCS = {"count", "sum", "avg", "min", "max"}
def _jaccard_sets(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def _is_ident(tok: str) -> bool:
    # Refined identifier detection
    if not tok:
        return False
    if tok in RESERVED_TOKENS:
        return False
    if tok in {MASK_STR, MASK_NUM}:
        return False
    if tok in _ONECHAR_OPS or tok in _TWOCHAR_OPS or tok in _PARENS or tok in _PUNCT:
        return False
    return ("." in tok) or tok[0].isalpha()

def _expr_signature(expr_str):
    toks = list(tokenize_ra_expression(expr_str))
    agg = toks[0] if toks and toks[0] in _AGG_FUNCS else None
    has_distinct = "distinct" in toks
    ids = [t for t in toks if _is_ident(t) and t not in _AGG_FUNCS and t != "distinct"]
    return {"agg": agg, "has_distinct": has_distinct, "ids": ids}

def expr_similarity_expr(p, q):
    w_agg, w_dist, w_ids = 0.6, 0.1, 0.3
    agg_sim = 1.0 if p["agg"] == q["agg"] else 0.0
    dist_sim = 1.0 if p["has_distinct"] == q["has_distinct"] else 0.0
    ids_sim = _jaccard_sets(p["ids"], q["ids"])
    return w_agg * agg_sim + w_dist * dist_sim + w_ids * ids_sim

def score_projection(pred_node, gt_node, debug=False):
    P_exprs = _ensure_list_and_get_exprs(pred_node.extra_info.get("Expressions"))
    G_exprs = _ensure_list_and_get_exprs(gt_node.extra_info.get("Expressions"))
    P = [_expr_signature(e) for e in P_exprs]
    G = [_expr_signature(e) for e in G_exprs]
    if not P and not G: score = 1.0
    elif not G: score = 0.0
    else:
        _, total = _best_greedy_bipartite_match(P, G, expr_similarity_expr)
        score = total / max(len(G), 1)
    if debug:
        print(f"[DEBUG][{gt_node.path}] PROJECTION")
        for tag, exprs, sigs in (("PRED", P_exprs, P), ("GOLD", G_exprs, G)):
            print(f"  {tag}_RAW: {exprs}")
            for e, s in zip(exprs, sigs):
                toks = tokenize_ra_expression(e)
                print(f"    expr: {e} | tokens: {toks} | sig: {s}")
        print(f"  PROJECTION_expressions score={score:.4f}")
    return score, [{"component": "PROJECTION_expressions", "score": score, "location": gt_node.path}]

_CMP_OPS = {"=", "==", "!=", "<>", "<", "<=", ">", ">=", "like"}
_COMMUTATIVE_EQ = {"=", "=="}

def _first_ident(tokens):
    # Prefer standard identifiers
    for t in tokens:
        if _is_ident(t):
            return t
    # Fallback: handle quoted identifiers (e.g., 'Educational Option Type')
    for t in tokens:
        if isinstance(t, str) and len(t) >= 2 and t[0] in ("'", '"', "`") and t[-1] == t[0]:
            inner = t[1:-1].strip()
            if inner:
                # Sanitize to an identifier-like form
                s = re.sub(r"\s+", "_", inner.lower())
                s = re.sub(r"[^a-z0-9_\.]", "_", s)
                if s:
                    return s
    return None

def _strip_commas_parens(tokens):
    return [t for t in tokens if t not in {",", "(", ")"}]

def _split_top_level_by(tokens, ops:set):
    depth = 0
    for i, t in enumerate(tokens):
        if t == "(": depth += 1
        elif t == ")": depth = max(0, depth - 1)
        elif depth == 0 and t in ops: return tokens[:i], t, tokens[i+1:]
    return None

def _strip_outer_parens_str(s: str) -> str:
    # Remove only a single layer of outer parentheses if balanced
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        toks = _lex_ra(s)
        if toks and toks[0] == "(" and toks[-1] == ")" and _balanced_parens(toks):
            s = s[1:-1].strip()
        else:
            break
    return s

def _fuse_multiword_identifier(ts):
    """
    Merge multi-word identifiers following a dotted prefix into a single token.
    Example: ['frpm.school', 'type'] -> ['frpm.school type']
             ['frpm.free', 'meal', 'count', '(', 'ages', '5', '-', '17', ')'] ->
             ['frpm.free meal count ( ages 5 - 17 )']
    Stops at operators, punctuation, reserved keywords (asc/desc), or parentheses not part of the name.
    """
    out = []
    i, n = 0, len(ts)
    while i < n:
        t = ts[i]
        if isinstance(t, str) and "." in t:
            parts = [t]
            j = i + 1
            # Append following name words until operator or ASC/DESC/reserved at top level
            while j < n:
                tok = ts[j]
                if tok == "(":
                    # Attach a single balanced (...) group as part of the name
                    d, k = 1, j + 1
                    while k < n and d > 0:
                        if ts[k] == "(":
                            d += 1
                        elif ts[k] == ")":
                            d -= 1
                        parts.append(ts[k])
                        k += 1
                    j = k
                    break  # stop after attaching one (...) group
                elif tok == ")":
                    # Do not consume a closing paren; it's not part of the identifier
                    break
                if tok in _ONECHAR_OPS or tok in _TWOCHAR_OPS or tok in _PUNCT or tok in {"asc", "desc"} or tok in RESERVED_TOKENS:
                    break
                parts.append(tok)
                j += 1
            out.append(" ".join(parts))
            i = j
            continue
        out.append(t)
        i += 1
    return tuple(out)

def _normalize_expr_string(s):
    if s is None:
        return None
    s = _strip_outer_parens_str(str(s))
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s

def _literal_from_cast(ts):
    """
    If tokens represent CAST('<string>') at top-level, return the quoted string token unchanged.
    This preserves content like (Y/N) without sanitization.
    """
    ts = list(ts)
    if len(ts) >= 4 and ts[0] == "cast" and ts[1] == "(" and ts[-1] == ")" and _balanced_parens(ts[1:]):
        inner = ts[2:-1]
        if len(inner) == 1 and isinstance(inner[0], str) and len(inner[0]) >= 2 and inner[0][0] in ("'", '"', "`") and inner[0][-1] == inner[0][0]:
            return inner[0]
    return None

def _is_func_call_tokens(ts):
    """
    Return True if tokens form a top-level function call like SUBSTRING(x,1,2).
    """
    ts = list(ts)
    return (
        len(ts) >= 3
        and isinstance(ts[0], str)
        and ts[1] == "("
        and ts[-1] == ")"
        and _balanced_parens(ts[1:])
    )

def parse_predicate(expr_str):
    # First, try to capture IS [NOT] NULL using raw string to preserve casing and full expression
    raw_s = str(expr_str).strip()
    raw_s = _strip_outer_parens_str(raw_s)

    m_not_is_null = re.match(r"^\s*NOT\s+(?P<lhs>.+?)\s+IS\s+NULL\s*$", raw_s, flags=re.IGNORECASE)
    m_is_not_null = re.match(r"^\s*(?P<lhs>.+?)\s+IS\s+NOT\s+NULL\s*$", raw_s, flags=re.IGNORECASE)
    m_is_null     = re.match(r"^\s*(?P<lhs>.+?)\s+IS\s+NULL\s*$", raw_s, flags=re.IGNORECASE)

    if m_not_is_null or m_is_not_null or m_is_null:
        kind = "is_not_null" if (m_not_is_null or m_is_not_null) else "is_null"
        lhs_raw = (m_not_is_null or m_is_not_null or m_is_null).group("lhs").strip()
        # Tokenize LHS without masking/canonicalization for debug/inspection
        lhs_tokens = list(tokenize_ra_expression(lhs_raw, mask_literals=False, canonicalize=False))
        lhs_tokens_fused = list(_fuse_multiword_identifier(lhs_tokens))
        # Use the full LHS expression for is_null/is_not_null
        return {
            "kind": kind,
            "lhs": lhs_raw,
            "op": None,
            "rhs": None,
            "rhs_list": None,
            "range": None,
            "lhs_expr": lhs_raw,
            "lhs_tokens": lhs_tokens_fused
        }

    # Do not mask literals and do not canonicalize equals while parsing for other predicate types
    toks = list(tokenize_ra_expression(expr_str, mask_literals=False, canonicalize=False))
    if len(toks) >= 2 and toks[0] == "(" and toks[-1] == ")" and _balanced_parens(toks):
        toks = toks[1:-1]
    toks = list(_fuse_multiword_identifier(toks))

    # Normalize "NOT <expr> IS NULL" -> "<expr> IS NOT NULL"
    if toks and toks[0] == "not":
        if "is_null" in toks:
            toks = toks[1:]
            idx = toks.index("is_null")
            toks[idx] = "is_not_null"

    if "is_not_null" in toks:
        idx = toks.index("is_not_null")
        lhs_slice = list(_fuse_multiword_identifier(toks[:idx]))
        lhs_expr = " ".join(lhs_slice)
        return {
            "kind": "is_not_null", "lhs": lhs_expr, "op": None, "rhs": None,
            "rhs_list": None, "range": None,
            "lhs_tokens": lhs_slice,
            "lhs_expr": lhs_expr
        }
    if "is_null" in toks:
        idx = toks.index("is_null")
        lhs_slice = list(_fuse_multiword_identifier(toks[:idx]))
        lhs_expr = " ".join(lhs_slice)
        return {
            "kind": "is_null", "lhs": lhs_expr, "op": None, "rhs": None,
            "rhs_list": None, "range": None,
            "lhs_tokens": lhs_slice,
            "lhs_expr": lhs_expr
        }

    op = "not_in" if "not_in" in toks else ("in" if "in" in toks else None)
    if op:
        idx = toks.index(op)
        lhs_slice = list(_fuse_multiword_identifier(toks[:idx]))
        lhs = _first_ident(lhs_slice)
        rhs_tokens = _strip_commas_parens(toks[idx+1:])
        return {"kind": op, "lhs": lhs, "op": op, "rhs": None, "rhs_list": rhs_tokens, "range": None}

    if "between" in toks and "and" in toks:
        b = toks.index("between"); depth = 0; and_pos = None
        for i in range(b+1, len(toks)):
            t = toks[i]
            if t == "(": depth += 1
            elif t == ")": depth = max(0, depth - 1)
            elif t == "and" and depth == 0: and_pos = i; break
        lhs_slice = list(_fuse_multiword_identifier(toks[:b]))
        lhs = _first_ident(lhs_slice)
        lo = " ".join(toks[b+1:and_pos]) if and_pos else None
        hi = " ".join(toks[and_pos+1:]) if and_pos else None
        return {"kind": "between", "lhs": lhs, "op": "between", "rhs": None, "rhs_list": None, "range": (lo, hi)}

    if "like" in toks:
        split = _split_top_level_by(toks, {"like"})
        if split:
            L, op, R = split
            L = list(_fuse_multiword_identifier(L)); R = list(_fuse_multiword_identifier(R))
            return {"kind": "like", "lhs": _first_ident(L), "op": op, "rhs": _first_ident(R) or " ".join(R), "rhs_list": None, "range": None}

    split = _split_top_level_by(toks, _CMP_OPS)
    if split:
        L, op, R = split
        L = list(_fuse_multiword_identifier(L)); R = list(_fuse_multiword_identifier(R))
        # Preserve CAST('...') as a literal on either side
        lhs_lit = _literal_from_cast(L)
        rhs_lit = _literal_from_cast(R)

        # LHS: prefer CAST literal; else if function-call, keep full expression; else use first identifier
        if lhs_lit is not None:
            lhs = lhs_lit
        elif _is_func_call_tokens(L):
            lhs = " ".join(L)
        else:
            lhs = _first_ident(L)

        # RHS: if function-call, keep full expression; else prefer identifier; else CAST literal or raw join
        if _is_func_call_tokens(R):
            rhs = " ".join(R)
            rhs_ident = None
        else:
            rhs_ident = _first_ident(R)
            rhs = (
                rhs_ident if rhs_ident is not None
                else (rhs_lit if rhs_lit is not None else (" ".join(R) if R else None))
            )

        # Prefer identifier on LHS for equality when LHS is not an identifier but RHS is
        if op in _COMMUTATIVE_EQ and (lhs is None or not _is_ident(lhs)) and (rhs_ident is not None):
            lhs = rhs_ident
            rhs = " ".join(L) if L else rhs

        return {"kind": "cmp", "lhs": lhs, "op": op, "rhs": rhs, "rhs_list": None, "range": None}

    lhs = _first_ident(list(_fuse_multiword_identifier(toks)))
    return {"kind": "cmp", "lhs": lhs, "op": None, "rhs": " ".join(toks), "rhs_list": None, "range": None}

def _interval_iou(a_lo, a_hi, b_lo, b_hi):
    def _try_float(x):
        try: return float(x)
        except Exception: return None
    fa, fb, fc, fd = _try_float(a_lo), _try_float(a_hi), _try_float(b_lo), _try_float(b_hi)
    if None in (fa, fb, fc, fd): return 1.0 if (a_lo and a_hi and b_lo and b_hi) else 0.0
    lo = max(min(fa, fb), min(fc, fd)); hi = min(max(fa, fb), max(fc, fd))
    inter = max(0.0, hi - lo)
    union = max(fa, fb) - min(fa, fb) + max(fc, fd) - min(fc, fd) - inter
    return 0.0 if union <= 0 else inter/union

def predicate_similarity(p, q):
    if p["kind"] != q["kind"]: return 0.0
    w_col, w_op, w_val = 0.6, 0.25, 0.15

    def col_sim(p, q):
        # For IS NULL / IS NOT NULL, compare full lhs expressions (normalized)
        if p["kind"] in {"is_null", "is_not_null"}:
            p_expr = p.get("lhs_expr") or p.get("lhs")
            q_expr = q.get("lhs_expr") or q.get("lhs")
            if p_expr is None or q_expr is None:
                return 0.0
            return 1.0 if _normalize_expr_string(p_expr) == _normalize_expr_string(q_expr) else 0.0

        if p["kind"] == "cmp" and p.get("op") in _COMMUTATIVE_EQ and q.get("op") in _COMMUTATIVE_EQ:
            cols_p = set([p["lhs"]] if p["lhs"] else [])
            if isinstance(p["rhs"], str) and _is_ident(p["rhs"]): cols_p.add(p["rhs"])
            cols_q = set([q["lhs"]] if q["lhs"] else [])
            if isinstance(q["rhs"], str) and _is_ident(q["rhs"]): cols_q.add(q["rhs"])
            return 1.0 if cols_p and cols_p == cols_q else (0.5 if cols_p & cols_q else 0.0)
        return 1.0 if (p["lhs"] is not None and p["lhs"] == q["lhs"]) else 0.0

    def op_sim(p, q):
        if p["kind"] in {"is_null", "is_not_null", "in", "not_in", "between", "like"}: return 1.0
        if p["op"] == q["op"]: return 1.0
        if {p["op"], q["op"]} <= {"!=", "<>"} or {p["op"], q["op"]} <= {"=", "=="}: return 1.0
        return 0.0

    def val_sim(p, q):
        if p["kind"] == "cmp" and p.get("op") in _COMMUTATIVE_EQ and q.get("op") in _COMMUTATIVE_EQ:
            sides_p = tuple(sorted([s for s in (p.get("lhs"), p.get("rhs")) if isinstance(s, str)]))
            sides_q = tuple(sorted([s for s in (q.get("lhs"), q.get("rhs")) if isinstance(s, str)]))
            if sides_p and sides_q and sides_p == sides_q: return 1.0
        if p["kind"] in {"is_null", "is_not_null"}: return 1.0
        if p["kind"] in {"in", "not_in"}:
            set_p, set_q = set(p["rhs_list"] or []), set(q["rhs_list"] or [])
            if not set_p and not set_q: return 1.0
            if not set_p or not set_q: return 0.0
            return len(set_p & set_q) / len(set_p | set_q)
        if p["kind"] == "between":
            lo_p, hi_p = (p["range"] or (None, None)); lo_q, hi_q = (q["range"] or (None, None))
            return _interval_iou(lo_p, hi_p, lo_q, hi_q)
        if p["kind"] == "like": return 1.0 if (p["rhs"] and q["rhs"]) else 0.0
        if isinstance(p["rhs"], str) and isinstance(q["rhs"], str):
            return 1.0 if p["rhs"] == q["rhs"] else 0.0
        return 0.0

    return w_col*col_sim(p,q) + w_op*op_sim(p,q) + w_val*val_sim(p,q)

def _best_greedy_bipartite_match(A, B, sim_fn):
    pairs = sorted([(i, j, sim_fn(a, b)) for i, a in enumerate(A) for j, b in enumerate(B)], key=lambda x: (-x[2], x[0], x[1]))
    used_i, used_j = set(), set(); matches, total = [], 0.0
    for i, j, s in pairs:
        if i in used_i or j in used_j: continue
        used_i.add(i); used_j.add(j); matches.append((i, j, s)); total += s
    return matches, total

def _parse_conjuncts(raw_expr_val):
    # More robust extraction from dict or stringified dict/list
    expr_str = ""
    if isinstance(raw_expr_val, dict) and "expr" in raw_expr_val:
        expr_str = raw_expr_val["expr"]
    elif isinstance(raw_expr_val, str):
        s = raw_expr_val.strip()
        if s and s[0] in "{[":
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict) and "expr" in parsed:
                    expr_str = parsed["expr"]
                else:
                    expr_str = raw_expr_val
            except Exception:
                expr_str = raw_expr_val
        else:
            expr_str = raw_expr_val
    return [parse_predicate(c) for c in split_conjuncts(expr_str)]

def score_filter(pred_node, gt_node, debug=False, use_f1_for_predicates=False):
    pred_raw = pred_node.extra_info.get("Filters")
    gold_raw = gt_node.extra_info.get("Filters")
    P = _parse_conjuncts(pred_raw)
    G = _parse_conjuncts(gold_raw)
    if not P and not G:
        score = 1.0
    elif not G:
        score = 0.0
    else:
        matches, total = _best_greedy_bipartite_match(P, G, predicate_similarity)
        if use_f1_for_predicates:
            prec = total / max(len(P), 1)
            rec  = total / max(len(G), 1)
            score = (2 * prec * rec / (prec + rec)) if (prec > 0 and rec > 0) else 0.0
        else:
            score = total / max(len(G), 1)
    if debug:
        print(f"[DEBUG][{gt_node.path}] FILTER")
        print(f"  PRED_FILTER_RAW: {pred_raw}")
        print(f"  GOLD_FILTER_RAW: {gold_raw}")
        def _dbg_pred_list(label, raw):
            if raw is None: print(f"  {label}_conjuncts: []"); return
            conjuncts = split_conjuncts(raw)
            print(f"  {label}_conjuncts ({len(conjuncts)}):")
            for c in conjuncts:
                print(f"    '{c}' | tokens={tokenize_ra_expression(c)}")
        _dbg_pred_list("PRED", pred_raw)
        _dbg_pred_list("GOLD", gold_raw)
        print("  Parsed predicates (kind,lhs,op,rhs,etc.):")
        for tag, arr in (("PRED", P), ("GOLD", G)):
            for i, p in enumerate(arr):
                print(f"    {tag}[{i}]: {p}")
        mode = "F1" if use_f1_for_predicates else "Recall"
        print(f"  FILTER_conditions ({mode}) score={score:.4f}")
    return score, [{"component": "FILTER_conditions", "score": score, "location": gt_node.path}]

def score_join(pred_node, gt_node, debug=False, use_f1_for_predicates=False):
    pred_raw = pred_node.extra_info.get("Condition")
    gold_raw = gt_node.extra_info.get("Condition")
    P = _parse_conjuncts(pred_raw)
    G = _parse_conjuncts(gold_raw)
    if not P and not G:
        cond_score = 1.0
    elif not G:
        cond_score = 0.0
    else:
        matches, total = _best_greedy_bipartite_match(P, G, predicate_similarity)
        if use_f1_for_predicates:
            prec = total / max(len(P), 1)
            rec  = total / max(len(G), 1)
            cond_score = (2 * prec * rec / (prec + rec)) if (prec > 0 and rec > 0) else 0.0
        else:
            cond_score = total / max(len(G), 1)
    gt_type = (gt_node.extra_info or {}).get("JoinType")
    pr_type = (pred_node.extra_info or {}).get("JoinType")
    type_score = 1.0 if gt_type == pr_type else 0.0
    score = 0.8 * cond_score + 0.2 * type_score
    if debug:
        print(f"[DEBUG][{gt_node.path}] JOIN")
        print(f"  PRED_JOIN_RAW: {pred_raw}")
        print(f"  GOLD_JOIN_RAW: {gold_raw}")
        def _dbg(raw, label):
            if raw is None: return
            conjuncts = split_conjuncts(raw)
            print(f"  {label}_conjuncts ({len(conjuncts)}):")
            for c in conjuncts:
                print(f"    '{c}' | tokens={tokenize_ra_expression(c)}")
        _dbg(pred_raw, "PRED")
        _dbg(gold_raw, "GOLD")
        for tag, arr in (("PRED", P), ("GOLD", G)):
            for i, p in enumerate(arr):
                print(f"    {tag}[{i}]: {p}")
        mode = "F1" if use_f1_for_predicates else "Recall"
        print(f"  JoinType PRED={pr_type} GOLD={gt_type} (JOIN_type score={type_score:.4f})")
        print(f"  JOIN_conditions ({mode}) score={cond_score:.4f}  Combined score={score:.4f}")
    return score, [
        {"component": "JOIN_conditions", "score": cond_score, "location": gt_node.path},
        {"component": "JOIN_type", "score": type_score, "location": gt_node.path}
    ]

def score_aggregate(pred_node, gt_node, debug=False):
    pred_expr_sigs = [_expr_signature(e) for e in _ensure_list_and_get_exprs(pred_node.extra_info.get("Expressions"))]
    gt_expr_sigs   = [_expr_signature(e) for e in _ensure_list_and_get_exprs(gt_node.extra_info.get("Expressions"))]
    if not pred_expr_sigs and not gt_expr_sigs: expr_score = 1.0
    elif not gt_expr_sigs: expr_score = 0.0
    else:
        _, total_expr = _best_greedy_bipartite_match(pred_expr_sigs, gt_expr_sigs, expr_similarity_expr)
        expr_score = total_expr / max(len(gt_expr_sigs), 1)
    pred_group_toks = {tokenize_ra_expression(g) for g in _ensure_list_and_get_exprs(pred_node.extra_info.get("Groups"))}
    gt_group_toks   = {tokenize_ra_expression(g) for g in _ensure_list_and_get_exprs(gt_node.extra_info.get("Groups"))}
    group_score = jaccard_similarity_on_token_tuples(pred_group_toks, gt_group_toks)
    score = 0.7 * expr_score + 0.3 * group_score
    if debug:
        print(f"[DEBUG][{gt_node.path}] AGGREGATE")
        print("  PRED expressions/signatures:")
        for raw in _ensure_list_and_get_exprs(pred_node.extra_info.get("Expressions")):
            print(f"    {raw} | tokens={tokenize_ra_expression(raw)} | sig={_expr_signature(raw)}")
        print("  GOLD expressions/signatures:")
        for raw in _ensure_list_and_get_exprs(gt_node.extra_info.get("Expressions")):
            print(f"    {raw} | tokens={tokenize_ra_expression(raw)} | sig={_expr_signature(raw)}")
        print(f"  PRED groups: {pred_group_toks}")
        print(f"  GOLD groups: {gt_group_toks}")
        print(f"  AGGREGATE_expressions score={expr_score:.4f}")
        print(f"  AGGREGATE_grouping score={group_score:.4f}  Combined={score:.4f}")
    return score, [
        {"component": "AGGREGATE_expressions", "score": expr_score, "location": gt_node.path},
        {"component": "AGGREGATE_grouping", "score": group_score, "location": gt_node.path}
    ]

def score_sort(pred_node, gt_node, debug=False):
    def _unwrap_cast_if_top_level(ts):
        # Repeatedly unwrap CAST(...) only when it wraps the entire expression
        changed = True
        ts = list(ts)
        while changed:
            changed = False
            # Peel outer parens if the whole seq is wrapped
            if len(ts) >= 2 and ts[0] == "(" and ts[-1] == ")" and _balanced_parens(ts):
                ts = ts[1:-1]
                changed = True
                continue
            # Unwrap top-level CAST(...)
            if len(ts) >= 3 and ts[0] == "cast" and ts[1] == "(" and ts[-1] == ")" and _balanced_parens(ts[1:]):
                ts = ts[2:-1]
                changed = True
        return tuple(ts)

    def _omit_cast_when_operand(ts):
        # Omit CAST(...) when it appears as a single top-level operand, e.g., CAST(x) / y
        ts = list(ts)
        changed = True
        while changed:
            changed = False
            depth = 0
            i = 0
            while i < len(ts):
                t = ts[i]
                if t == "(":
                    depth += 1
                elif t == ")":
                    depth = max(0, depth - 1)
                elif depth == 0 and t == "cast" and i + 1 < len(ts) and ts[i + 1] == "(":
                    # find matching ')'
                    d = 1
                    j = i + 2
                    while j < len(ts) and d > 0:
                        if ts[j] == "(":
                            d += 1
                        elif ts[j] == ")":
                            d -= 1
                        j += 1
                    if d == 0:
                        close_idx = j - 1
                        prev_ok = (i == 0) or (ts[i - 1] in {"+", "-", "*", "/", "(", ","})
                        next_ok = (close_idx == len(ts) - 1) or (ts[close_idx + 1] in {"+", "-", "*", "/", ")", ",", "asc", "desc"})
                        if prev_ok and next_ok:
                            inner = ts[i + 2:close_idx]
                            ts = ts[:i] + inner + ts[close_idx + 1:]
                            changed = True
                            depth = 0
                            i = -1  # restart scan
                i += 1
        return tuple(ts)

    # Note: use module-level _fuse_multiword_identifier

    pred_raw_orders = _ensure_list_and_get_exprs(pred_node.extra_info.get("Order"))
    gt_raw_orders   = _ensure_list_and_get_exprs(gt_node.extra_info.get("Order"))
    # Do NOT mask literals (numbers or strings) in ORDER BY expressions
    pred_toks = {tokenize_ra_expression(o, mask_literals=False) for o in pred_raw_orders}
    gt_toks   = {tokenize_ra_expression(o, mask_literals=False) for o in gt_raw_orders}

    # Normalize for comparison:
    # 1) fuse multi-word identifiers after dotted prefix
    # 2) unwrap only top-level CAST and redundant parens
    # 3) additionally omit CAST when used as a direct operand at top-level
    pred_toks = {_fuse_multiword_identifier(ts) for ts in pred_toks}
    gt_toks   = {_fuse_multiword_identifier(ts) for ts in gt_toks}
    pred_toks_norm = {_omit_cast_when_operand(_unwrap_cast_if_top_level(ts)) for ts in pred_toks}
    gt_toks_norm   = {_omit_cast_when_operand(_unwrap_cast_if_top_level(ts)) for ts in gt_toks}

    order_score = jaccard_similarity_on_token_tuples(pred_toks_norm, gt_toks_norm)

    pred_limit_str = pred_node.extra_info.get("Limit"); gt_limit_str = gt_node.extra_info.get("Limit")
    try:
        pred_limit = int(pred_limit_str) if pred_limit_str is not None else None
        gt_limit   = int(gt_limit_str) if gt_limit_str is not None else None
        if gt_limit is None and pred_limit is None: limit_score = 1.0
        elif gt_limit is None or pred_limit is None: limit_score = 0.0
        elif gt_limit == 0: limit_score = 1.0 if pred_limit == 0 else 0.0
        else: limit_score = max(0.0, 1.0 - abs(pred_limit - gt_limit) / gt_limit)
    except (ValueError, TypeError):
        limit_score = 1.0 if str(pred_limit_str) == str(gt_limit_str) else 0.0

    score = 0.8 * order_score + 0.2 * limit_score
    if debug:
        print(f"[DEBUG][{gt_node.path}] SORT")
        print(f"  PRED order tokens: {pred_toks}")
        print(f"  GOLD order tokens: {gt_toks}")
        print(f"  PRED order tokens (norm): {pred_toks_norm}")
        print(f"  GOLD order tokens (norm): {gt_toks_norm}")
        print(f"  Limits PRED={pred_limit_str} GOLD={gt_limit_str}")
        print(f"  SORT_orderby score={order_score:.4f} SORT_limit score={limit_score:.4f} Combined={score:.4f}")
    return score, [
        {"component": "SORT_orderby", "score": order_score, "location": gt_node.path},
        {"component": "SORT_limit", "score": limit_score, "location": gt_node.path}
    ]

def score_window(pred_node, gt_node, debug=False):
    pred_toks = {tokenize_ra_expression(e) for e in _ensure_list_and_get_exprs(pred_node.extra_info.get("Expressions"))}
    gt_toks   = {tokenize_ra_expression(e) for e in _ensure_list_and_get_exprs(gt_node.extra_info.get("Expressions"))}
    score = jaccard_similarity_on_token_tuples(pred_toks, gt_toks)
    if debug:
        print(f"[DEBUG][{gt_node.path}] WINDOW")
        print(f"  PRED expr tokens: {pred_toks}")
        print(f"  GOLD expr tokens: {gt_toks}")
        print(f"  WINDOW_expressions score={score:.4f}")
    return score, [{"component": "WINDOW_expressions", "location": gt_node.path, "score": score}]

def score_distinct(pred_node, gt_node, debug=False):
    if debug:
        print(f"[DEBUG][{gt_node.path}] DISTINCT (always score 1.0)")
    return 1.0, [{"component": "DISTINCT_operator", "score": 1.0, "location": gt_node.path}]

def score_default(pred_node, gt_node, debug=False):
    score = 1.0 if pred_node.extra_info.get("Table") == gt_node.extra_info.get("Table") else 0.0
    if debug:
        print(f"[DEBUG][{gt_node.path}] DEFAULT/TABLE")
        print(f"  PRED Table={pred_node.extra_info.get('Table')} GOLD Table={gt_node.extra_info.get('Table')} -> TABLE_match={score:.4f}")
    return score, [{"component": "TABLE_match", "score": score, "location": gt_node.path}]

# NEW: Component weights for unified recall score
COMPONENT_WEIGHTS = {
    "PROJECTION_expressions": 1.0, "FILTER_conditions": 1.0,
    "JOIN_conditions": 0.8, "JOIN_type": 0.2,
    "AGGREGATE_expressions": 0.7, "AGGREGATE_grouping": 0.3,
    "SORT_orderby": 0.8, "SORT_limit": 0.2,
    "WINDOW_expressions": 1.0, "DISTINCT_operator": 1.0,
    "TABLE_match": 1.0, "OPERATOR_MATCH": 1.0
}

def _evaluate_recursive(predicted_ra, ground_truth_ra, debug=False):
    if not predicted_ra and not ground_truth_ra:
        return {"components": [], "weighted_score": 0.0, "total_weight": 0.0}
    if not predicted_ra or not ground_truth_ra:
        total_weight = get_total_weight(ground_truth_ra or predicted_ra)
        return {"components": [], "weighted_score": 0.0, "total_weight": total_weight}

    score_func_map = {
        "PROJECTION": score_projection,
        "FILTER": score_filter,
        "JOIN": score_join,
        "COMPARISON_JOIN": score_join,
        "AGGREGATE": score_aggregate,
        "SORT": score_sort,
        "WINDOW": score_window,
        "DISTINCT": score_distinct
    }
    all_components = []
    node_weighted_score = 0.0
    node_total_weight = 0.0

    if debug:
        print(f"[DEBUG] Evaluating node path={ground_truth_ra.path} GT_OP={ground_truth_ra.name} PRED_OP={predicted_ra.name}")

    if predicted_ra.name == ground_truth_ra.name:
        all_components.append({"component": "OPERATOR_MATCH", "score": 1.0, "location": ground_truth_ra.path})
        score_func = score_func_map.get(predicted_ra.name, score_default)
        # Call with debug flag
        _, component_scores = score_func(predicted_ra, ground_truth_ra, debug=debug)
        all_components.extend(component_scores)
    else:
        all_components.append({"component": "OPERATOR_MATCH", "score": 0.0, "location": ground_truth_ra.path})
        if debug:
            print(f"[DEBUG][{ground_truth_ra.path}] Operator mismatch: PRED={predicted_ra.name} GOLD={ground_truth_ra.name}")

    for comp in all_components:
        weight = COMPONENT_WEIGHTS.get(comp["component"], 0.0)
        node_weighted_score += comp["score"] * weight
        node_total_weight += weight

    if len(predicted_ra.children) != len(ground_truth_ra.children):
        all_components.append({"component": "child_structure_mismatch", "score": 0.0, "location": ground_truth_ra.path})
        # When child counts differ, we still need to add the expected weights of all GT child subtrees
        missing_weight_total = 0.0
        for gt_child in ground_truth_ra.children:
            child_w = get_total_weight(gt_child)   # get_total_weight returns a float (subtree total)
            node_total_weight += child_w
            missing_weight_total += child_w
        if debug:
            print(f"[DEBUG][{ground_truth_ra.path}] Child count mismatch "
                  f"PRED={len(predicted_ra.children)} GOLD={len(ground_truth_ra.children)} "
                  f"=> added expected subtree weight={missing_weight_total:.4f}")
    else:
        all_components.append({"component": "child_structure_mismatch", "score": 1.0, "location": ground_truth_ra.path})
        for pred_child, gt_child in zip(predicted_ra.children, ground_truth_ra.children):
            child_eval = _evaluate_recursive(pred_child, gt_child, debug=debug)
            all_components.extend(child_eval["components"])
            node_weighted_score += child_eval["weighted_score"]
            node_total_weight += child_eval["total_weight"]

    return {"components": all_components, "weighted_score": node_weighted_score, "total_weight": node_total_weight}

def get_total_weight(gt_node):
    if gt_node is None: return 0.0
    
    node_weight = COMPONENT_WEIGHTS.get(f"{gt_node.name}_expressions", 1.0) # Default
    if gt_node.name == "JOIN" or gt_node.name == "COMPARISON_JOIN":
        node_weight = COMPONENT_WEIGHTS["JOIN_conditions"] + COMPONENT_WEIGHTS["JOIN_type"]
    elif gt_node.name == "AGGREGATE":
        node_weight = COMPONENT_WEIGHTS["AGGREGATE_expressions"] + COMPONENT_WEIGHTS["AGGREGATE_grouping"]
    elif gt_node.name == "SORT":
        node_weight = COMPONENT_WEIGHTS["SORT_orderby"] + COMPONENT_WEIGHTS["SORT_limit"]
    
    total_weight = node_weight + COMPONENT_WEIGHTS["OPERATOR_MATCH"]
    
    for child in gt_node.children:
        total_weight += get_total_weight(child)
        
    return total_weight

def evaluate(predicted_ra, ground_truth_ra, debug=False, allow_child_scoring_on_mismatch=False, use_f1_for_predicates=False):
    if not predicted_ra and not ground_truth_ra:
        return {"score": 1.0, "components": [], "component_recall_score": 1.0}
    if not predicted_ra or not ground_truth_ra:
        return {"score": 0.0, "components": [], "component_recall_score": 0.0}

    eval_result = _evaluate_recursive(predicted_ra, ground_truth_ra, debug=debug)
    total_weight = get_total_weight(ground_truth_ra)

    # Content-aware tree score: blend node component score with child structural scores
    score_func_map = {
        "PROJECTION": score_projection,
        "FILTER": score_filter,
        "JOIN": score_join,
        "COMPARISON_JOIN": score_join,
        "AGGREGATE": score_aggregate,
        "SORT": score_sort,
        "WINDOW": score_window,
        "DISTINCT": score_distinct
    }

    def component_tree_score(p, g, alpha=0.5):
        if not p and not g: return 1.0
        if not p or not g: return 0.0
        if p.name != g.name:
            if allow_child_scoring_on_mismatch:
                node_score = 0.0
                if len(p.children) != len(g.children):
                    child_avg = 0.0
                else:
                    child_scores = [component_tree_score(pc, gc, alpha) for pc, gc in zip(p.children, g.children)]
                    child_avg = sum(child_scores) / len(child_scores) if child_scores else 1.0
                return alpha * node_score + (1 - alpha) * child_avg
            return 0.0
        sf = score_func_map.get(p.name, score_default)
        if sf in (score_filter, score_join):
            node_score, _ = sf(p, g, debug=False, use_f1_for_predicates=use_f1_for_predicates)
        else:
            node_score, _ = sf(p, g, debug=False)
        if len(p.children) != len(g.children):
            child_avg = 0.0
        else:
            child_scores = [component_tree_score(pc, gc, alpha) for pc, gc in zip(p.children, g.children)]
            child_avg = sum(child_scores) / len(child_scores) if child_scores else 1.0
        return alpha * node_score + (1 - alpha) * child_avg

    tree_score = component_tree_score(predicted_ra, ground_truth_ra, alpha=0.5)
    recall_score = eval_result["weighted_score"] / total_weight if total_weight > 0 else 0.0
    if debug:
        print(f"[DEBUG] Final tree_score={tree_score:.4f} component_recall={recall_score:.4f}")
    return {
        "score": tree_score,
        "components": eval_result["components"],
        "component_recall_score": recall_score
    }

# =======================
# Convenience wrappers
# =====================

# do not change the function below, as it is used externally
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



def ra_eval_res(ground_truth_ra_json, predicted_ra_json, debug=False, allow_child_scoring_on_mismatch=False, use_f1_for_predicates=False):
    ground_truth_tree = parse_ra(ground_truth_ra_json)
    predicted_tree    = parse_ra(predicted_ra_json)
    return evaluate(predicted_tree, ground_truth_tree, debug=debug, allow_child_scoring_on_mismatch=allow_child_scoring_on_mismatch, use_f1_for_predicates=use_f1_for_predicates)

def print_eval_res(ground_truth_ra_json, predicted_ra_json, debug=False, allow_child_scoring_on_mismatch=False, use_f1_for_predicates=False):
    ground_truth_tree = parse_ra(ground_truth_ra_json)
    predicted_tree    = parse_ra(predicted_ra_json)
    evaluation_result = evaluate(predicted_tree, ground_truth_tree, debug=debug, allow_child_scoring_on_mismatch=allow_child_scoring_on_mismatch, use_f1_for_predicates=use_f1_for_predicates)

    print("\n--- Relational Algebra Evaluation Report ---")
    print(f"\n  Overall Tree Score: {evaluation_result['score']:.2f}")
    print(f"  Component Recall Score: {evaluation_result['component_recall_score']:.2f}")

    print("\n  Component Score Breakdown:")
    # Sort for consistent output
    sorted_components = sorted(evaluation_result['components'], key=lambda x: (x['location'], x['component']))
    for item in sorted_components:
        print(f"    - Location: {item['location']}")
        print(f"      Component: {item['component']}")
        print(f"      Score: {item['score']:.2f}")

    return evaluation_result