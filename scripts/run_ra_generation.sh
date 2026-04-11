#!/bin/bash
# Generate relational algebra trees from SQL queries using Apache Calcite.
#
# Prerequisites:
#   1. Build the Calcite JAR: cd ra_generation/calcite-plan && mvn clean package
#   2. Place dataset files under ra_generation/data/ (see README for details)
#   3. Generate per-database model.json files (--generate-models flag)

cd "$(dirname "$0")/../ra_generation"

# Step 1: Generate model.json files for each database
python ra_gen.py --datasets spider_dev --generate-models

# Step 2: Run RA generation
python ra_gen.py --datasets spider_dev

# For multiple datasets:
# python ra_gen.py --datasets spider_train spider_dev bird_train bird_dev

# For SynSQL (uses parallel processing):
# python ra_gen.py --datasets synsql --generate-models
