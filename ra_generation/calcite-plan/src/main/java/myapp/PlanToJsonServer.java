package myapp;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class PlanToJsonServer {
  public static void main(String[] args) throws Exception {
    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    String line;
    while ((line = br.readLine()) != null) {
      String[] parts = line.split("\\|", 2);
      String model = parts[0];
      String sql = parts[1];

      // System.out.println("Received model:");
      // System.out.println(model);
      // System.out.println("Received SQL:");
      // System.out.println(sql);

      try {
        String json = PlanToJson.runOnce(model, sql);
        System.out.println(json);
      } catch (Exception e) {
        System.err.println("ERROR: " + e.getMessage());
      }
      System.out.flush();
    }
  }
}