package myapp;

// PlanPretty.java
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.*;
import java.util.*;
import java.io.*;

public class PlanPretty {
  // Usage: java PlanPretty < plan.json
  public static void main(String[] args) throws Exception {
    ObjectMapper M = new ObjectMapper();
    JsonNode root = M.readTree(System.in);

    // The RelJson format has "rels" (operators) and an index graph.
    ArrayNode rels = (ArrayNode) root.get("rels");
    Map<Integer, JsonNode> byId = new HashMap<>();
    for (int i = 0; i < rels.size(); i++) {
      JsonNode n = rels.get(i);
      byId.put(n.get("id").asInt(), n);
    }

    // Find root (the last rel is typically the root)
    JsonNode top = rels.get(rels.size() - 1);
    printNode(top, byId, 0);
  }

  static void printNode(JsonNode node, Map<Integer, JsonNode> byId, int depth) {
    String op = node.get("op").asText();
    List<String> fields = getFields(node);

    // Compose header line
    String header = op;
    // Annotate common attributes (condition, group, order)
    if (node.has("condition")) {
      header += "(condition=" + expr(node.get("condition"), node, byId) + ")";
    }
    if (node.has("group")) {
      header += header.contains("(") ? ", " : "(";
      header += "group=" + fieldListFromIdx(node.get("group"), fields) + ")";
    }
    if (node.has("order")) {
      header += header.contains("(") ? ", " : "(";
      header += "order=" + orderByNames(node.get("order"), fields) + ")";
    }

    indent(depth);
    System.out.println(header);

    // Recurse into inputs
    ArrayNode inputs = (ArrayNode) node.get("inputs");
    if (inputs != null) {
      for (JsonNode in : inputs) {
        JsonNode child = byId.get(in.asInt());
        printNode(child, byId, depth + 1);
      }
    }
  }

  // Map field indexes to names at this node
  static List<String> getFields(JsonNode node) {
    ArrayNode f = (ArrayNode) node.get("fields");
    List<String> out = new ArrayList<>();
    if (f != null) for (JsonNode x : f) out.add(x.asText());
    return out;
  }

  static String fieldListFromIdx(JsonNode arr, List<String> fields) {
    if (arr == null || !arr.isArray()) return "[]";
    List<String> out = new ArrayList<>();
    for (JsonNode i : arr) out.add(safe(fields, i.asInt()));
    return out.toString();
  }

  // Render ORDER BY entries with names
  static String orderByNames(JsonNode arr, List<String> fields) {
    if (arr == null || !arr.isArray()) return "[]";
    List<String> out = new ArrayList<>();
    for (JsonNode e : arr) {
      int idx = e.get("field").asInt();
      String dir = e.has("dir") ? e.get("dir").asText() : "ASC";
      out.add(safe(fields, idx) + " " + dir);
    }
    return out.toString();
  }

  // Render expressions (joins/filters). RelJson uses structured nodes.
  static String expr(JsonNode e, JsonNode node, Map<Integer, JsonNode> byId) {
    if (e == null) return "";
    if (e.has("input")) {
      int idx = e.get("input").asInt();
      List<String> fields = getFields(node);
      return safe(fields, idx);
    }
    if (e.has("literal")) return e.get("literal").toString();
    if (e.has("op")) {
      String op = e.get("op").asText();
      ArrayNode args = (ArrayNode) e.get("operands");
      List<String> parts = new ArrayList<>();
      for (JsonNode a : args) parts.add(expr(a, node, byId));
      // infix for common ops
      if (parts.size() == 2 && Arrays.asList("=","<>",">","<",">=","<=","AND","OR").contains(op))
        return "(" + parts.get(0) + " " + op + " " + parts.get(1) + ")";
      // function style
      return op + "(" + String.join(", ", parts) + ")";
    }
    // Fallback
    return e.toString();
  }

  static void indent(int d) {
    for (int i = 0; i < d; i++) System.out.print("  ");
  }

  static String safe(List<String> fields, int i) {
    if (i >= 0 && i < fields.size()) return fields.get(i);
    return "$" + i;
  }
}
