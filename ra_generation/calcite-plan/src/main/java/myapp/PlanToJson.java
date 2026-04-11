package myapp;

import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.validate.SqlConformanceEnum;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;

import com.fasterxml.jackson.databind.ObjectMapper;

import org.apache.calcite.schema.SchemaPlus;

import java.sql.Connection;
import java.sql.DriverManager;
import java.util.Properties;

public class PlanToJson {
  // Usage: java PlanToJson <model.json> "<SQL>"

  public static String runOnce(String model, String sql) throws Exception {

    Properties info = new Properties();
    info.setProperty("caseSensitive", "false");
    info.setProperty("model", model);

    try (Connection conn = DriverManager.getConnection("jdbc:calcite:", info)) {
      CalciteConnection calcite = conn.unwrap(CalciteConnection.class);
      SchemaPlus root = calcite.getRootSchema();

      // If your model has multiple top-level schemas, pick one. Adjust as needed.
      // For your example it sounded like "sqlite" is present:
      SchemaPlus defaultSchema = root.getSubSchema("sqlite");
      if (defaultSchema == null) {
        // fall back to root if not found
        defaultSchema = root;
      }

      // System.out.println("Using default schema: " + defaultSchema.getName());
      // for (String tableName : defaultSchema.getTableNames()) {
      // System.out.println("Table: " + tableName);
      // System.out.println(" Columns: " + defaultSchema.getTable(tableName)
      // .getRowType(calcite.getTypeFactory()).getFieldNames());
      // }

      SqlParser.Config parserConfig = SqlParser.config()
          .withUnquotedCasing(Casing.UNCHANGED)
          .withCaseSensitive(false)
          // Be a bit more forgiving for SQLite-ish SQL
          .withConformance(SqlConformanceEnum.LENIENT);

      FrameworkConfig cfg = Frameworks.newConfigBuilder()
          .defaultSchema(defaultSchema)
          .parserConfig(parserConfig)
          .build();

      Planner planner = Frameworks.getPlanner(cfg);

      // 1) parse & validate
      SqlNode parsed = planner.parse(sql);
      SqlNode validated = planner.validate(parsed);

      // 2) to RelNode
      RelNode rel = planner.rel(validated).rel;

      // System.out.println("Relational algebra plan:");
      // System.out.println(rel.explain());

      // 3) to JSON (custom writer)
      JsonRelWriter writer = new JsonRelWriter();
      ObjectMapper mapper = new ObjectMapper();
      String json = mapper.writerWithDefaultPrettyPrinter()
          .writeValueAsString(writer.explain(rel));

      return json;
    } catch (Exception e) {
      // e.printStackTrace();
      // return e.getMessage() + "error\": \"Failed to generate JSON plan";
      // throw the exception to be handled by the caller
      throw e;
    }
  }

  public static void main(String[] args) throws Exception {
    final String model;
    final String sql;

    if (args.length >= 2) {
      model = args[0];
      sql = args[1];
    } else {
      // // Fallback sample for convenience
      model = "./model_template.json";
      sql = "SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1";
      System.err.println("Usage: java PlanToJson <model.json> \"<SQL>\"");
      System.err.println("Using fallback model and SQL:");
      System.err.println("Model: " + model);
      System.err.println("SQL: " + sql);
    }

    String json = runOnce(model, sql);
    System.out.println(json);
  }
}
