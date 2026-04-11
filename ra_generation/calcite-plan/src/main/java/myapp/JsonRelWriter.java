package myapp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.*;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rel.metadata.RelColumnOrigin;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.util.NlsString;
import org.apache.calcite.util.DateString;
import org.apache.calcite.util.TimeString;
import org.apache.calcite.util.TimestampString;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class JsonRelWriter {
    private final ObjectMapper mapper = new ObjectMapper();

    /** Entry */
    public ObjectNode explain(RelNode rel) {
        ObjectNode root = mapper.createObjectNode();
        root.set("relational_algebra", explainNode(rel));
        return root;
    }

    /** Recursive tree walk */
    private ObjectNode explainNode(RelNode rel) {
        ObjectNode node = mapper.createObjectNode();
        node.put("name", getOpName(rel));

        ObjectNode extra = mapper.createObjectNode();

        // === Generic TableScan (covers LogicalTableScan & JdbcTableScan etc.) ===
        if (rel instanceof TableScan) {
            TableScan scan = (TableScan) rel;
            String tableName = lastSafe(scan.getTable().getQualifiedName());
            if (tableName != null)
                extra.put("Table", tableName);
            extra.put("Type", "Sequential Scan");
        }

        // === Filter ===
        if (rel instanceof LogicalFilter) {
            LogicalFilter f = (LogicalFilter) rel;
            List<String> qNames = qualifiedNamesForOutput(f.getInput());
            Map<Integer, String> idxMap = mapFromList(qNames);
            extra.set("Filters", buildExprNode(f.getCondition(), f, idxMap));
        }

        // === Project ===
        if (rel instanceof LogicalProject) {
            LogicalProject p = (LogicalProject) rel;

            // qualified names for the *input* of the project
            List<String> inQNames = qualifiedNamesForOutput(p.getInput());
            Map<Integer, String> inQIdxMap = mapFromList(inQNames);

            ArrayNode exprs = mapper.createArrayNode();
            for (int i = 0; i < p.getProjects().size(); i++) {
                RexNode e = p.getProjects().get(i);

                String exprStr;
                ArrayNode cols = mapper.createArrayNode();

                if (e instanceof RexInputRef && p.getInput() instanceof LogicalAggregate) {
                    // Project over Aggregate: resolve grouping vs agg output explicitly
                    LogicalAggregate agg = (LogicalAggregate) p.getInput();
                    int outIdx = ((RexInputRef) e).getIndex();

                    if (outIdx < agg.getGroupCount()) {
                        // group key -> use qualified input column name
                        int inputIdx = agg.getGroupSet().asList().get(outIdx);
                        exprStr = qualifiedNamesForOutput(agg.getInput()).get(inputIdx);
                        cols.add(exprStr);
                    } else {
                        // aggregate output -> pretty print expr, and columns from agg args (not $fN)
                        int aggIdx = outIdx - agg.getGroupCount();
                        AggregateCall ac = agg.getAggCallList().get(aggIdx);

                        List<String> qInputNames = qualifiedNamesForOutput(agg.getInput());
                        exprStr = prettyAggRef(ac, qInputNames);

                        if (!ac.getArgList().isEmpty()) {
                            for (Integer a : ac.getArgList()) {
                                cols.add(qInputNames.get(a));
                            }
                        }
                        // COUNT(*) -> columns stays []
                        // other aggs with no args -> also []
                    }
                } else {
                    // Normal case: format expr and collect qualified input refs
                    exprStr = formatRexSQL(e, inQIdxMap);
                    LinkedHashSet<String> names = new LinkedHashSet<>();
                    collectInputRefs(e, inQIdxMap, names);
                    for (String n : names)
                        cols.add(n);
                }

                ObjectNode exprNode = mapper.createObjectNode();
                exprNode.put("expr", exprStr);
                exprNode.set("columns", cols);
                exprs.add(exprNode);
            }
            extra.set("Expressions", exprs);
        }

        // === Join ===
        if (rel instanceof LogicalJoin) {
            LogicalJoin j = (LogicalJoin) rel;
            List<String> leftQ = qualifiedNamesForOutput(j.getLeft());
            List<String> rightQ = qualifiedNamesForOutput(j.getRight());
            Map<Integer, String> joinMap = concatMaps(mapFromList(leftQ), mapFromList(rightQ));

            extra.set("Condition", buildExprNode(j.getCondition(), j, joinMap));
            extra.put("JoinType", j.getJoinType().name());
        }

        // === Aggregate ===
        if (rel instanceof LogicalAggregate) {
            LogicalAggregate g = (LogicalAggregate) rel;
            List<String> inQNames = qualifiedNamesForOutput(g.getInput());

            // Groups as ARRAY of qualified names
            ArrayNode groupArr = mapper.createArrayNode();
            g.getGroupSet().asList().forEach(idx -> {
                if (idx >= 0 && idx < inQNames.size())
                    groupArr.add(inQNames.get(idx));
                else
                    groupArr.add("?" + idx);
            });
            extra.set("Groups", groupArr);

            // Aggregates
            ArrayNode aggs = mapper.createArrayNode();
            Map<Integer, String> idxMap = mapFromList(inQNames);

            g.getAggCallList().forEach(a -> {
                ObjectNode exprNode = mapper.createObjectNode();
                if (a.getArgList().isEmpty()) {
                    exprNode.put("expr", a.getAggregation().getName() + "(*)");
                    exprNode.set("columns", mapper.createArrayNode());
                } else {
                    String args = a.getArgList().stream()
                            .map(i -> idxMap.getOrDefault(i, "?" + i))
                            .collect(Collectors.joining(", "));
                    exprNode.put("expr", a.getAggregation().getName() + "(" + args + ")");
                    ArrayNode cols = mapper.createArrayNode();
                    a.getArgList().forEach(i -> cols.add(idxMap.getOrDefault(i, "?" + i)));
                    exprNode.set("columns", cols);
                }
                aggs.add(exprNode);
            });

            // Handle DISTINCT (no agg calls), special case in Calcite: SELECT DISTINCT a,b
            if (g.getAggCallList().isEmpty() && !g.getGroupSet().isEmpty()) {
                ObjectNode exprNode = mapper.createObjectNode();
                exprNode.put("expr", "DISTINCT");
                exprNode.set("columns", mapper.createArrayNode()); // empty array
                aggs.add(exprNode);
            }

            extra.set("Expressions", aggs);
        }

        // === Sort (ORDER BY + LIMIT/OFFSET) ===
        if (rel instanceof LogicalSort) {
            LogicalSort s = (LogicalSort) rel;
            ArrayNode order = mapper.createArrayNode();
            for (RelFieldCollation fc : s.getCollation().getFieldCollations()) {
                String col = prettyNameForOrderItem(s.getInput(), fc.getFieldIndex());
                String dir = fc.getDirection().toString().contains("DESC") ? "DESC" : "ASC";

                // Optional: suppress NULLS ... entirely (you already removed it)
                order.add((col + " " + dir).trim());
            }
            if (order.size() > 0)
                extra.set("Order", order);

            // LIMIT
            if (s.fetch != null) {
                if (s.fetch instanceof RexLiteral) {
                    Object v = ((RexLiteral) s.fetch).getValue2();
                    if (v instanceof Number)
                        extra.put("Limit", ((Number) v).intValue());
                } else {
                    List<String> inQNames = qualifiedNamesForOutput(s.getInput());
                    extra.put("LimitExpr", formatRexSQL(s.fetch, mapFromList(inQNames)));
                }
            }
            // OFFSET
            if (s.offset != null) {
                if (s.offset instanceof RexLiteral) {
                    Object v = ((RexLiteral) s.offset).getValue2();
                    if (v instanceof Number)
                        extra.put("Offset", ((Number) v).intValue());
                } else {
                    List<String> inQNames = qualifiedNamesForOutput(s.getInput());
                    extra.put("OffsetExpr", formatRexSQL(s.offset, mapFromList(inQNames)));
                }
            }
        }

        // === Set ops: include ALL/DISTINCT ===
        if (rel instanceof LogicalUnion) {
            extra.put("All", ((LogicalUnion) rel).all);
        } else if (rel instanceof LogicalIntersect) {
            extra.put("All", ((LogicalIntersect) rel).all);
        } else if (rel instanceof LogicalMinus) {
            extra.put("All", ((LogicalMinus) rel).all);
        }

        // === Window functions (basic) ===
        if (rel instanceof LogicalWindow) {
            LogicalWindow w = (LogicalWindow) rel;
            ArrayNode groups = mapper.createArrayNode();
            List<String> inQNames = qualifiedNamesForOutput(w.getInput());
            Map<Integer, String> idxMap = mapFromList(inQNames);

            for (org.apache.calcite.rel.core.Window.Group g : w.groups) {
                ObjectNode gnode = mapper.createObjectNode();

                // PARTITION BY (qualified)
                ArrayNode part = mapper.createArrayNode();
                g.keys.asList().forEach(i -> part.add(idxMap.getOrDefault(i, "?" + i)));
                if (part.size() > 0)
                    gnode.set("PartitionBy", part);

                // ORDER BY (qualified)
                ArrayNode ob = mapper.createArrayNode();
                for (RelFieldCollation fc : g.orderKeys.getFieldCollations()) {
                    String col = idxMap.getOrDefault(fc.getFieldIndex(), "?" + fc.getFieldIndex());
                    String dir = fc.getDirection().toString().contains("DESC") ? "DESC" : "ASC";
                    ob.add(col + " " + dir);
                }
                if (ob.size() > 0)
                    gnode.set("OrderBy", ob);

                gnode.put("Rows", g.isRows);
                gnode.put("LowerBound", String.valueOf(g.lowerBound));
                gnode.put("UpperBound", String.valueOf(g.upperBound));

                // Windowed agg calls
                ArrayNode calls = mapper.createArrayNode();
                for (org.apache.calcite.rel.core.Window.RexWinAggCall c : g.aggCalls) {
                    ObjectNode cnode = mapper.createObjectNode();
                    cnode.put("function", c.getOperator().getName());
                    ArrayNode args = mapper.createArrayNode();
                    for (RexNode a : c.getOperands()) {
                        args.add(formatRexSQL(a, idxMap));
                    }
                    cnode.set("args", args);
                    calls.add(cnode);
                }
                if (calls.size() > 0)
                    gnode.set("Calls", calls);

                groups.add(gnode);
            }
            if (groups.size() > 0)
                extra.set("WindowGroups", groups);
        }

        node.set("extra_info", extra);

        // children
        ArrayNode children = mapper.createArrayNode();
        for (RelNode input : rel.getInputs()) {
            children.add(explainNode(input));
        }
        node.set("children", children);

        return node;
    }

    /* ---------- Expression node ---------- */
    private ObjectNode buildExprNode(RexNode rex, RelNode rel, Map<Integer, String> idxToName) {
        ObjectNode exprNode = mapper.createObjectNode();
        RelMetadataQuery mq = rel.getCluster().getMetadataQuery();

        // Subquery
        if (rex instanceof RexSubQuery) {
            RexSubQuery sub = (RexSubQuery) rex;

            exprNode.put("expr", sub.getKind().toString());
            exprNode.set("subquery", explainNode(sub.rel));

            if (sub.getOperands() != null && !sub.getOperands().isEmpty()) {
                ArrayNode ops = mapper.createArrayNode();
                for (RexNode o : sub.getOperands()) {
                    ops.add(formatRexSQL(o, idxToName));
                }
                exprNode.set("operands", ops);

                String leftCols = StreamSupport.stream(ops.spliterator(), false)
                        .map(JsonNode::asText)
                        .collect(Collectors.joining(", "));
                exprNode.put("expr", leftCols + " " + sub.getKind().toString() + " (subquery)");
            }
            return exprNode;
        }

        // Column reference (with origin tracking; QUALIFIED)
        if (rex instanceof RexInputRef) {
            int idx = ((RexInputRef) rex).getIndex();
            ArrayNode cols = mapper.createArrayNode();

            Set<RelColumnOrigin> origins = mq.getColumnOrigins(rel, idx);
            if (origins != null && !origins.isEmpty()) {
                for (RelColumnOrigin co : origins) {
                    String table = (co.getOriginTable() != null)
                            ? lastSafe(co.getOriginTable().getQualifiedName())
                            : null;
                    String col = (co.getOriginTable() != null)
                            ? co.getOriginTable().getRowType().getFieldList()
                                    .get(co.getOriginColumnOrdinal()).getName()
                            : idxToName.getOrDefault(idx, "?" + idx);
                    cols.add((table != null ? table + "." : "") + col);
                }
                String expr = (cols.size() == 1) ? cols.get(0).asText()
                        : idxToName.getOrDefault(idx, "?" + idx);
                exprNode.put("expr", expr);
                exprNode.set("columns", cols);
                if (cols.size() > 1)
                    exprNode.put("disambiguated", true);
                return exprNode;
            }

            // fallback: derived column
            String name = idxToName.getOrDefault(idx, "?" + idx);
            exprNode.put("expr", name);
            cols.add(name);
            exprNode.set("columns", cols);
            return exprNode;
        }

        // Other expressions (literals / calls)
        exprNode.put("expr", formatRexSQL(rex, idxToName));

        // Collect referenced columns directly from the expression (qualified via
        // idxToName)
        ArrayNode cols = mapper.createArrayNode();
        LinkedHashSet<String> names = new LinkedHashSet<>();
        collectInputRefs(rex, idxToName, names);
        for (String n : names)
            cols.add(n);
        exprNode.set("columns", cols);

        return exprNode;
    }

    /* ---------- Helpers ---------- */

    /** Prefer qualified names table.column for each output field of this rel. */
    private List<String> qualifiedNamesForOutput(RelNode rel) {
        List<String> out = new ArrayList<>();
        RelMetadataQuery mq = rel.getCluster().getMetadataQuery();
        List<String> fieldNames = rel.getRowType().getFieldNames();
        for (int i = 0; i < fieldNames.size(); i++) {
            String fallback = fieldNames.get(i);
            String qualified = fallback;

            try {
                Set<RelColumnOrigin> origins = mq.getColumnOrigins(rel, i);
                if (origins != null && !origins.isEmpty()) {
                    // pick the first origin deterministically
                    RelColumnOrigin co = origins.iterator().next();
                    if (co.getOriginTable() != null) {
                        String table = lastSafe(co.getOriginTable().getQualifiedName());
                        String col = co.getOriginTable().getRowType().getFieldList()
                                .get(co.getOriginColumnOrdinal()).getName();
                        qualified = (table != null ? table + "." : "") + col;
                    }
                }
            } catch (Throwable ignore) {
                // fall back to field name if metadata isn't available
            }

            out.add(qualified);
        }
        return out;
    }

    private static Map<Integer, String> mapFromList(List<String> names) {
        Map<Integer, String> m = new HashMap<>();
        for (int i = 0; i < names.size(); i++)
            m.put(i, names.get(i));
        return m;
    }

    private static Map<Integer, String> concatMaps(Map<Integer, String> left, Map<Integer, String> right) {
        Map<Integer, String> m = new HashMap<>(left);
        int offset = left.size();
        right.forEach((k, v) -> m.put(k + offset, v));
        return m;
    }

    private static String lastSafe(List<String> qn) {
        return (qn == null || qn.isEmpty()) ? null : qn.get(qn.size() - 1);
    }

    private static String prettyAggRef(AggregateCall ac, List<String> inputNames) {
        String aggName = ac.getAggregation().getName();
        if (ac.getArgList().isEmpty()) {
            return aggName.equalsIgnoreCase("COUNT") ? "count_star" : (aggName + "(*)");
        }
        List<String> args = ac.getArgList().stream()
                .map(inputNames::get)
                .collect(Collectors.toList());
        return aggName + "(" + String.join(", ", args) + ")";
    }

    /** Collect input references (QUALIFIED via idxToName). */
    private void collectInputRefs(RexNode node, Map<Integer, String> idxToName, LinkedHashSet<String> out) {
        if (node == null)
            return;
        if (node instanceof RexInputRef) {
            int idx = ((RexInputRef) node).getIndex();
            out.add(idxToName.getOrDefault(idx, "?" + idx));
            return;
        }
        if (node instanceof RexCall) {
            RexCall c = (RexCall) node;
            for (RexNode o : c.getOperands())
                collectInputRefs(o, idxToName, out);
            return;
        }
        if (node instanceof RexSubQuery) {
            RexSubQuery sq = (RexSubQuery) node;
            if (sq.getOperands() != null) {
                for (RexNode o : sq.getOperands())
                    collectInputRefs(o, idxToName, out);
            }
        }
        // literals / others: no-op
    }

    /** Pretty column name for ORDER BY position, avoiding $fN/EXPR$N; QUALIFIED. */
    private String prettyNameForOrderItem(RelNode input, int fieldIndex) {
        if (input instanceof LogicalProject) {
            LogicalProject proj = (LogicalProject) input;
            RexNode expr = proj.getProjects().get(fieldIndex);

            if (expr instanceof RexInputRef && proj.getInput() instanceof LogicalAggregate) {
                LogicalAggregate agg = (LogicalAggregate) proj.getInput();
                int idx = ((RexInputRef) expr).getIndex();
                if (idx < agg.getGroupCount()) {
                    int inputIdx = agg.getGroupSet().asList().get(idx);
                    return qualifiedNamesForOutput(agg.getInput()).get(inputIdx);
                } else {
                    int aggIdx = idx - agg.getGroupCount();
                    AggregateCall ac = agg.getAggCallList().get(aggIdx);
                    return prettyAggRef(ac, qualifiedNamesForOutput(agg.getInput()));
                }
            }

            return formatRexSQL(expr, mapFromList(qualifiedNamesForOutput(proj.getInput())));
        }

        List<String> names = qualifiedNamesForOutput(input);
        if (fieldIndex >= 0 && fieldIndex < names.size()) {
            String n = names.get(fieldIndex);
            if ((n.startsWith("$f") || n.startsWith("EXPR$")) && !input.getInputs().isEmpty()) {
                return prettyNameForOrderItem(input.getInput(0), fieldIndex);
            }
            return n;
        }
        return "$f" + fieldIndex;
    }

    /** SQL-ish printer for Rex */
    private String formatRexSQL(RexNode rex, Map<Integer, String> idxToName) {
        if (rex instanceof RexInputRef) {
            int idx = ((RexInputRef) rex).getIndex();
            return idxToName.getOrDefault(idx, "?" + idx);
        } else if (rex instanceof RexLiteral) {
            RexLiteral lit = (RexLiteral) rex;
            if (lit.getValue() == null)
                return "NULL";

            Object v = lit.getValue2();
            if (v instanceof NlsString) {
                return "'" + ((NlsString) v).getValue().replace("'", "''") + "'";
            } else if (v instanceof String) {
                return "'" + ((String) v).replace("'", "''") + "'";
            } else if (v instanceof DateString) {
                return "DATE '" + ((DateString) v).toString() + "'";
            } else if (v instanceof TimeString) {
                return "TIME '" + ((TimeString) v).toString() + "'";
            } else if (v instanceof TimestampString) {
                return "TIMESTAMP '" + ((TimestampString) v).toString() + "'";
            } else {
                return String.valueOf(v);
            }

        } else if (rex instanceof RexCall) {
            RexCall c = (RexCall) rex;
            SqlKind k = c.getKind();

            // Common binary infix
            if (c.getOperands().size() == 2 && (k == SqlKind.EQUALS ||
                    k == SqlKind.NOT_EQUALS ||
                    k == SqlKind.GREATER_THAN ||
                    k == SqlKind.GREATER_THAN_OR_EQUAL ||
                    k == SqlKind.LESS_THAN ||
                    k == SqlKind.LESS_THAN_OR_EQUAL ||
                    k == SqlKind.AND ||
                    k == SqlKind.OR ||
                    k == SqlKind.PLUS ||
                    k == SqlKind.MINUS ||
                    k == SqlKind.TIMES ||
                    k == SqlKind.DIVIDE ||
                    k == SqlKind.LIKE)) {
                String l = formatRexSQL(c.getOperands().get(0), idxToName);
                String r = formatRexSQL(c.getOperands().get(1), idxToName);
                return l + " " + c.getOperator().getName() + " " + r;
            }

            // BETWEEN a AND b (operands: value, lower, upper)
            if (k == SqlKind.BETWEEN && c.getOperands().size() == 3) {
                String v = formatRexSQL(c.getOperands().get(0), idxToName);
                String lo = formatRexSQL(c.getOperands().get(1), idxToName);
                String hi = formatRexSQL(c.getOperands().get(2), idxToName);
                return v + " BETWEEN " + lo + " AND " + hi;
            }

            // IN list: IN(x, a, b, c) -> x IN (a,b,c)
            if (k == SqlKind.IN && c.getOperands().size() >= 2) {
                String left = formatRexSQL(c.getOperands().get(0), idxToName);
                List<String> rest = c.getOperands().subList(1, c.getOperands().size())
                        .stream().map(o -> formatRexSQL(o, idxToName)).collect(Collectors.toList());
                return left + " IN (" + String.join(", ", rest) + ")";
            }

            // Unary: NOT, IS [NOT] NULL, +x, -x
            if (c.getOperands().size() == 1 &&
                    (k == SqlKind.NOT ||
                            k == SqlKind.IS_NULL || k == SqlKind.IS_NOT_NULL ||
                            k == SqlKind.PLUS_PREFIX || k == SqlKind.MINUS_PREFIX)) {

                RexNode op0 = c.getOperands().get(0);
                String a = formatRexSQL(op0, idxToName);

                switch (k) {
                    case NOT:
                        return "NOT " + a; // prefix
                    case IS_NULL:
                        return a + " IS NULL";
                    case IS_NOT_NULL:
                        return a + " IS NOT NULL";
                    case MINUS_PREFIX:
                        return "-" + a;
                    case PLUS_PREFIX:
                        return "+" + a;
                    default:
                        return c.getOperator().getName() + " " + a;
                }
            }

            // Fallback: function-style
            List<String> ops = c.getOperands().stream()
                    .map(o -> formatRexSQL(o, idxToName))
                    .collect(Collectors.toList());
            return c.getOperator().getName() + "(" + String.join(", ", ops) + ")";
        }

        // Fallback
        return rex.toString();
    }

    private String getOpName(RelNode rel) {
        if (rel instanceof LogicalProject)
            return "PROJECTION";
        if (rel instanceof LogicalFilter)
            return "FILTER";
        if (rel instanceof LogicalJoin)
            return "JOIN";
        if (rel instanceof LogicalAggregate)
            return "AGGREGATE";
        if (rel instanceof LogicalSort)
            return "SORT";
        if (rel instanceof TableScan)
            return "SEQ_SCAN";
        if (rel instanceof LogicalUnion)
            return "UNION";
        if (rel instanceof LogicalIntersect)
            return "INTERSECT";
        if (rel instanceof LogicalMinus)
            return "MINUS";
        if (rel instanceof LogicalValues)
            return "VALUES";
        if (rel instanceof LogicalCalc)
            return "CALC";
        if (rel instanceof LogicalExchange)
            return "EXCHANGE";
        if (rel instanceof LogicalWindow)
            return "WINDOW";
        return rel.getRelTypeName().toUpperCase();
    }
}
