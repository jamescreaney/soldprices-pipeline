# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Bronze: load a raw Price Paid file
# MAGIC
# MAGIC **What it does:** reads one HM Land Registry Price Paid CSV from the landing Volume and
# MAGIC writes it, unchanged, to the bronze table `workspace.default.bronze_price_paid`.
# MAGIC Each row also gets two load-metadata columns:
# MAGIC - `source_file`: which file the row came from
# MAGIC - `load_timestamp`: when it was loaded
# MAGIC
# MAGIC **Why:** bronze is a faithful copy of the source. If a later step has a bug, we can
# MAGIC rebuild silver and gold from here without downloading the file again.
# MAGIC
# MAGIC **Design choices**
# MAGIC - **Every column is loaded as text (STRING).** If we typed `price` as an integer here,
# MAGIC   a bad value would quietly become NULL and we'd lose what the file actually said.
# MAGIC   Typing and cleaning happen in silver, where we can see and count the failures.
# MAGIC - **Safe to re-run.** Running it twice for the same file replaces that file's rows
# MAGIC   rather than doubling them. Rows from other files are left alone, which matters once
# MAGIC   monthly update files arrive.
# MAGIC - **Fails loudly.** If the row counts don't reconcile or a basic check fails, the notebook
# MAGIC   raises an error, so a scheduled job shows as failed instead of passing silently.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings
# MAGIC Plain Python variables. Change `SOURCE_FILE` to load a different file.

# COMMAND ----------

SOURCE_FILE = "pp-2025.csv"
LANDING_PATH = "/Volumes/workspace/default/landing/"
BRONZE_TABLE = "workspace.default.bronze_price_paid"

source_path = LANDING_PATH + SOURCE_FILE
print(source_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define the schema
# MAGIC
# MAGIC **New concept: `StructType` / `StructField`.** This is the PySpark version of the column
# MAGIC list in a `CREATE TABLE`. Each `StructField` is one column: name, data type, and whether
# MAGIC NULLs are allowed (`True` = nullable).
# MAGIC
# MAGIC The file has no header row, so Spark can't know the column names. If we gave no schema,
# MAGIC it would call them `_c0`, `_c1`, ... and guess the types by scanning the data. An explicit
# MAGIC schema is faster and predictable.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

raw_schema = StructType([
    StructField("transaction_id",   StringType(), True),
    StructField("price",            StringType(), True),
    StructField("date_of_transfer", StringType(), True),
    StructField("postcode",         StringType(), True),
    StructField("property_type",    StringType(), True),
    StructField("old_new",          StringType(), True),
    StructField("duration",         StringType(), True),
    StructField("paon",             StringType(), True),
    StructField("saon",             StringType(), True),
    StructField("street",           StringType(), True),
    StructField("locality",         StringType(), True),
    StructField("town_city",        StringType(), True),
    StructField("district",         StringType(), True),
    StructField("county",           StringType(), True),
    StructField("ppd_category",     StringType(), True),
    StructField("record_status",    StringType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Read the CSV
# MAGIC
# MAGIC **New concept: DataFrame.** A DataFrame is a table-shaped result, like the output of a
# MAGIC `SELECT`. `spark.read...load()` gives a DataFrame over the file.
# MAGIC
# MAGIC **New concept: lazy evaluation.** This cell doesn't read any data yet. Spark only records
# MAGIC *how* to read it, much like defining a view. The work happens when something needs a
# MAGIC result, such as `.count()`, `display()` or writing to a table.
# MAGIC
# MAGIC Options:
# MAGIC - `header = false`: the first line is data, not column names.
# MAGIC - `mode = FAILFAST`: stop with an error if a line can't be parsed (for example, the
# MAGIC   wrong number of columns). The default mode would quietly fill the gaps with NULLs.
# MAGIC
# MAGIC The `( ... )` around the chain just lets us put one step on each line.

# COMMAND ----------

raw_df = (
    spark.read
    .format("csv")
    .schema(raw_schema)
    .option("header", "false")
    .option("mode", "FAILFAST")
    .load(source_path)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Add load metadata
# MAGIC
# MAGIC **New concept: `withColumn`.** Adds a column (or replaces one with the same name).
# MAGIC SQL equivalent: `SELECT *, <expression> AS new_column FROM ...`.
# MAGIC
# MAGIC **New concept: `F` (pyspark.sql.functions).** The library of column functions: the
# MAGIC equivalent of SQL's built-in functions. `F.col("x")` refers to column `x`, and
# MAGIC `F.current_timestamp()` is `CURRENT_TIMESTAMP()`.
# MAGIC
# MAGIC `_metadata` is a hidden column Databricks provides for any file read. It holds details
# MAGIC such as the file name and path. (Older tutorials use `input_file_name()`, which doesn't
# MAGIC work on Unity Catalog / serverless.)

# COMMAND ----------

from pyspark.sql import functions as F

bronze_df = (
    raw_df
    .withColumn("source_file", F.col("_metadata.file_name"))
    .withColumn("load_timestamp", F.current_timestamp())
)

display(bronze_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create the bronze table (first run only)
# MAGIC
# MAGIC Plain SQL, run from Python with `spark.sql(...)`. `IF NOT EXISTS` means later runs skip it.
# MAGIC
# MAGIC **New concept: f-string.** `f"""..."""` is a Python string where `{BRONZE_TABLE}` is
# MAGIC replaced by the variable's value, like a template. Tables in Databricks are **Delta**
# MAGIC tables by default: files plus a transaction log, which gives you ACID updates and history.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {BRONZE_TABLE} (
    transaction_id   STRING,
    price            STRING,
    date_of_transfer STRING,
    postcode         STRING,
    property_type    STRING,
    old_new          STRING,
    duration         STRING,
    paon             STRING,
    saon             STRING,
    street           STRING,
    locality         STRING,
    town_city        STRING,
    district         STRING,
    county           STRING,
    ppd_category     STRING,
    record_status    STRING,
    source_file      STRING,
    load_timestamp   TIMESTAMP
)
COMMENT 'Raw HM Land Registry Price Paid Data: one row per line of each source file, all columns as text. Typed and cleaned in silver.'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to bronze (safe to re-run)
# MAGIC
# MAGIC `mode("overwrite")` together with `replaceWhere` means: *replace only the rows where
# MAGIC `source_file` = this file, and leave the rest of the table alone.*
# MAGIC
# MAGIC SQL equivalent, done as **one atomic transaction**:
# MAGIC ```sql
# MAGIC BEGIN;
# MAGIC   DELETE FROM bronze_price_paid WHERE source_file = 'pp-2025.csv';
# MAGIC   INSERT INTO bronze_price_paid SELECT ... FROM <the file>;
# MAGIC COMMIT;
# MAGIC ```
# MAGIC As a safety net, Delta also refuses the write if any incoming row *doesn't* match the
# MAGIC `replaceWhere` condition.

# COMMAND ----------

(
    bronze_df.write
    .format("delta")
    .mode("overwrite")
    .option("replaceWhere", f"source_file = '{SOURCE_FILE}'")
    .saveAsTable(BRONZE_TABLE)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Row-count reconciliation
# MAGIC
# MAGIC Count the lines in the raw file (reading it as plain text, one row per line) and compare
# MAGIC with the rows now in bronze for this file. They should match exactly.
# MAGIC
# MAGIC `spark.table(...)` is `SELECT * FROM table`. `.filter(...)` is `WHERE`. `.count()` is
# MAGIC `COUNT(*)`, and it triggers the actual work (see "lazy evaluation" above).

# COMMAND ----------

file_line_count = spark.read.text(source_path).count()

bronze_this_file = spark.table(BRONZE_TABLE).filter(F.col("source_file") == SOURCE_FILE)
bronze_row_count = bronze_this_file.count()

print(f"Lines in file:          {file_line_count:,}")
print(f"Rows in bronze (file):  {bronze_row_count:,}")

if file_line_count != bronze_row_count:
    raise Exception(f"Reconciliation FAILED: file has {file_line_count:,} lines, bronze has {bronze_row_count:,} rows")

print("Reconciliation passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Basic data quality checks
# MAGIC
# MAGIC **New concept: `.agg(...)`.** Runs aggregate functions over the DataFrame. With no
# MAGIC `groupBy` first, the result is a single row, like `SELECT COUNT(*), MIN(...), ... FROM t`.
# MAGIC
# MAGIC `F.when(cond, 0).otherwise(1)` is `CASE WHEN cond THEN 0 ELSE 1 END`. Summing it counts
# MAGIC the rows that fail. A NULL `record_status` falls through to `otherwise`, so it's counted as bad.
# MAGIC
# MAGIC Checks that **fail** the notebook: missing transaction ids, missing prices, unknown
# MAGIC record_status. Duplicate transaction ids only produce a **warning**, because de-duplicating
# MAGIC is silver's job and we want to see how many there are first.

# COMMAND ----------

checks_df = bronze_this_file.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("transaction_id").alias("distinct_transaction_ids"),
    F.sum(F.when(F.col("transaction_id").isNotNull(), 0).otherwise(1)).alias("null_transaction_ids"),
    F.sum(F.when(F.col("price").isNotNull(), 0).otherwise(1)).alias("null_prices"),
    F.sum(F.when(F.col("record_status").isin("A", "C", "D"), 0).otherwise(1)).alias("bad_record_status"),
    F.min("date_of_transfer").alias("min_date"),
    F.max("date_of_transfer").alias("max_date"),
)

display(checks_df)

# .first() runs the query and returns the single result row; fields are read by name.
checks = checks_df.first()

problems = []
if checks["null_transaction_ids"] > 0:
    problems.append(f"{checks['null_transaction_ids']:,} rows with no transaction_id")
if checks["null_prices"] > 0:
    problems.append(f"{checks['null_prices']:,} rows with no price")
if checks["bad_record_status"] > 0:
    problems.append(f"{checks['bad_record_status']:,} rows with record_status not A/C/D")

duplicates = checks["row_count"] - checks["distinct_transaction_ids"]
if duplicates > 0:
    print(f"WARNING: {duplicates:,} duplicate transaction_id rows (handled in silver)")

if problems:
    raise Exception("Data quality checks FAILED: " + "; ".join(problems))

print("Data quality checks passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Have a look (SQL)
# MAGIC A `%sql` cell runs plain SQL against the same table.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT record_status, property_type, COUNT(*) AS row_count
# MAGIC FROM workspace.default.bronze_price_paid
# MAGIC WHERE source_file = 'pp-2025.csv'
# MAGIC GROUP BY record_status, property_type
# MAGIC ORDER BY record_status, property_type
