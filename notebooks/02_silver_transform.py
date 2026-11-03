# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Silver: clean, type and apply changes
# MAGIC
# MAGIC **What it does:** takes one source file's rows from bronze, then cleans and types them,
# MAGIC de-duplicates them, and **merges** them into `workspace.default.silver_price_paid`.
# MAGIC Silver holds one row per `transaction_id`, showing the latest known version of each sale.
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Read this file's rows from bronze
# MAGIC 2. Cast the types (price → number, date → DATE) and tidy the text columns
# MAGIC 3. Check nothing failed to cast; stop *before* writing anything if it did
# MAGIC 4. Standardise the postcode and add `postcode_district`
# MAGIC 5. Add readable labels (property type, tenure, new build) and flag PPD category B
# MAGIC 6. De-duplicate on `transaction_id` within the file
# MAGIC 7. MERGE into silver: **A** inserts, **C** updates, **D** deletes
# MAGIC 8. Reconcile the row counts: bronze → changes → silver
# MAGIC 9. Run quality checks on the finished silver table
# MAGIC
# MAGIC **Why a MERGE rather than rebuilding the table?** Land Registry's monthly update files only
# MAGIC contain what changed: new sales (A), corrected sales (C) and withdrawn sales (D). A MERGE
# MAGIC applies those changes to what's already there. For the yearly file every row is A, so
# MAGIC the first run is simply an insert.
# MAGIC
# MAGIC **Run files in date order.** Each run applies one file's changes on top of the current
# MAGIC silver table, so the last file run wins. Re-running the same file is safe: the result is
# MAGIC the same.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings

# COMMAND ----------

SOURCE_FILE = "pp-2025.csv"
BRONZE_TABLE = "workspace.default.bronze_price_paid"
SILVER_TABLE = "workspace.default.silver_price_paid"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Read this file's rows from bronze
# MAGIC SQL equivalent: `SELECT * FROM bronze_price_paid WHERE source_file = 'pp-2025.csv'`

# COMMAND ----------

from pyspark.sql import functions as F

bronze_file_df = spark.table(BRONZE_TABLE).filter(F.col("source_file") == SOURCE_FILE)

bronze_row_count = bronze_file_df.count()
print(f"Bronze rows for {SOURCE_FILE}: {bronze_row_count:,}")

if bronze_row_count == 0:
    raise Exception(f"No bronze rows for {SOURCE_FILE}. Run 01_bronze_load for it first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Cast the types and tidy the text columns
# MAGIC
# MAGIC **New concept: `.select(...)`.** This is SQL's `SELECT` list. You name every output column,
# MAGIC and `.alias("name")` is `AS name`. Any column not listed is dropped.
# MAGIC
# MAGIC **New concept: `F.expr("...")`.** This lets you write a piece of SQL inside PySpark. Where
# MAGIC SQL reads more clearly than the Python version, this notebook uses it. Both styles produce
# MAGIC exactly the same query plan.
# MAGIC
# MAGIC **Why `try_cast` and not `cast`?** Serverless runs in ANSI mode, where a plain `CAST('abc' AS BIGINT)`
# MAGIC stops the whole query with an error. `try_cast` returns NULL instead, so we can *count* the
# MAGIC failures in step 3 and report them. To make that possible, we keep the raw text alongside
# MAGIC the typed value for now (`price_raw`, `date_raw`).
# MAGIC
# MAGIC Dates arrive as `2025-01-15 00:00`. The time part is always midnight, so we take the first 10
# MAGIC characters and cast them to a DATE.
# MAGIC
# MAGIC **New concept: a Python function (`def`).** `clean_text` below is a small reusable
# MAGIC expression: trim the spaces and turn an empty string into NULL. SQL equivalent:
# MAGIC `NULLIF(TRIM(col), '') AS col`. It saves writing that out 14 times.

# COMMAND ----------

def clean_text(column_name):
    """Trim spaces and turn empty strings into NULL. Keeps the column name."""
    return F.expr(f"nullif(trim({column_name}), '')").alias(column_name)


typed_df = bronze_file_df.select(
    clean_text("transaction_id"),
    F.col("price").alias("price_raw"),
    F.expr("try_cast(trim(price) AS BIGINT)").alias("price"),
    F.col("date_of_transfer").alias("date_raw"),
    F.expr("try_cast(substr(trim(date_of_transfer), 1, 10) AS DATE)").alias("date_of_transfer"),
    clean_text("postcode"),
    clean_text("property_type"),
    clean_text("old_new"),
    clean_text("duration"),
    clean_text("paon"),
    clean_text("saon"),
    clean_text("street"),
    clean_text("locality"),
    clean_text("town_city"),
    clean_text("district"),
    clean_text("county"),
    clean_text("ppd_category"),
    clean_text("record_status"),
    F.col("source_file"),
    F.col("load_timestamp").alias("bronze_load_timestamp"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Check the casts, and stop before writing if anything is wrong
# MAGIC
# MAGIC A *cast failure* is a row that had a value in bronze but came out NULL after the cast.
# MAGIC These checks run **before** the MERGE, so bad data never reaches silver.
# MAGIC
# MAGIC `F.when(condition, 1).otherwise(0)` is `CASE WHEN condition THEN 1 ELSE 0 END`, and
# MAGIC summing it counts the rows that match. `&` means AND. (In PySpark you write `&`, `|` and `~`
# MAGIC for AND, OR and NOT, and each condition needs its own brackets.)

# COMMAND ----------

def count_where(condition):
    """SUM(CASE WHEN condition THEN 1 ELSE 0 END)"""
    return F.sum(F.when(condition, 1).otherwise(0))


pre_checks = typed_df.agg(
    count_where(F.col("transaction_id").isNull()).alias("null_transaction_ids"),
    count_where(F.col("price_raw").isNotNull() & F.col("price").isNull()).alias("price_cast_failures"),
    count_where(F.col("price").isNull()).alias("null_prices"),
    count_where(F.col("date_raw").isNotNull() & F.col("date_of_transfer").isNull()).alias("date_cast_failures"),
    count_where(F.col("date_of_transfer").isNull()).alias("null_dates"),
    count_where(~F.col("property_type").isin("D", "S", "T", "F", "O") | F.col("property_type").isNull()).alias("unknown_property_types"),
    count_where(~F.col("record_status").isin("A", "C", "D") | F.col("record_status").isNull()).alias("unknown_record_status"),
    F.min("date_of_transfer").alias("min_date"),
    F.max("date_of_transfer").alias("max_date"),
)
display(pre_checks)

c = pre_checks.first()

problems = []
for check_name in ["null_transaction_ids", "price_cast_failures", "null_prices",
                   "date_cast_failures", "null_dates", "unknown_property_types", "unknown_record_status"]:
    if c[check_name] > 0:
        problems.append(f"{c[check_name]:,} {check_name}")

if problems:
    raise Exception("Pre-merge checks FAILED, nothing written to silver: " + "; ".join(problems))

print("Pre-merge checks passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Standardise the postcode and add `postcode_district`
# MAGIC
# MAGIC Postcodes can arrive as `so23 1aa`, `SO231AA` or `SO23  1AA`. UK postcodes always end in a
# MAGIC 3-character inward code (digit + 2 letters), so the standard form is:
# MAGIC 1. Upper-case it and remove every space: `SO231AA`
# MAGIC 2. Put a single space before the last 3 characters: `SO23 1AA`
# MAGIC 3. Check it against the UK postcode pattern, and set `postcode_valid` true or false
# MAGIC 4. `postcode_district` is the outward code, the part before the space: `SO23`.
# MAGIC    It's only set when the postcode is valid.
# MAGIC
# MAGIC A postcode that can't be fixed is kept as it was (cleaned up) and flagged `postcode_valid = false`.
# MAGIC It isn't dropped: the sale is still real, it just can't be grouped by district.
# MAGIC Some sales have no postcode at all (often new-build plots).
# MAGIC
# MAGIC **New concept: `withColumn` replacing a column.** Using an existing name
# MAGIC (`"postcode"`) overwrites it, like `SELECT ..., <new expression> AS postcode`.
# MAGIC Each `withColumn` can use the columns created before it, which is how we build
# MAGIC the postcode step by step.
# MAGIC
# MAGIC `rlike` is a regular-expression match, like Snowflake's `RLIKE` / `REGEXP_LIKE`.

# COMMAND ----------

UK_POSTCODE_PATTERN = r"^[A-Z]{1,2}[0-9][A-Z0-9]? [0-9][A-Z]{2}$"

postcode_df = (
    typed_df
    .withColumn("postcode_compact", F.expr("nullif(upper(replace(postcode, ' ', '')), '')"))
    .withColumn("postcode", F.expr("""
        CASE
            WHEN length(postcode_compact) BETWEEN 5 AND 7
                THEN concat(left(postcode_compact, length(postcode_compact) - 3), ' ', right(postcode_compact, 3))
            ELSE postcode_compact
        END
    """))
    .withColumn("postcode_valid", F.coalesce(F.col("postcode").rlike(UK_POSTCODE_PATTERN), F.lit(False)))
    .withColumn("postcode_district", F.when(F.col("postcode_valid"), F.split(F.col("postcode"), " ").getItem(0)))
    .drop("postcode_compact")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Readable labels and the PPD category B flag
# MAGIC
# MAGIC `F.when(...).when(...).otherwise(...)` is a `CASE WHEN` chain. The original code columns are
# MAGIC kept next to the labels, so you can always trace a label back to the source value.
# MAGIC
# MAGIC | Column | Source codes | New column |
# MAGIC |---|---|---|
# MAGIC | property_type | D / S / T / F / O | `property_type_label`: Detached, Semi-detached, Terraced, Flat/Maisonette, Other |
# MAGIC | old_new | Y / N | `is_new_build`: true / false |
# MAGIC | duration | F / L | `duration_label`: Freehold / Leasehold (anything else is Unknown) |
# MAGIC | ppd_category | A / B | `is_additional_ppd`: true when B |
# MAGIC
# MAGIC **Why flag category B rather than drop it?** A is a standard sale at full market value. B is
# MAGIC "additional price paid", which mixes repossessions, buy-to-let purchases, transfers to companies
# MAGIC and other non-standard sales. The file has no way to tell these apart. Gold will usually
# MAGIC exclude B from average prices, but the data is kept, so that's a choice made in gold.

# COMMAND ----------

labelled_df = (
    postcode_df
    .withColumn("property_type_label",
        F.when(F.col("property_type") == "D", "Detached")
         .when(F.col("property_type") == "S", "Semi-detached")
         .when(F.col("property_type") == "T", "Terraced")
         .when(F.col("property_type") == "F", "Flat/Maisonette")
         .when(F.col("property_type") == "O", "Other"))
    .withColumn("is_new_build", F.col("old_new") == "Y")
    .withColumn("duration_label",
        F.when(F.col("duration") == "F", "Freehold")
         .when(F.col("duration") == "L", "Leasehold")
         .otherwise("Unknown"))
    .withColumn("is_additional_ppd", F.col("ppd_category") == "B")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: De-duplicate on `transaction_id` within this file
# MAGIC
# MAGIC **New concept: window functions.** These work exactly as in SQL. If a transaction appears
# MAGIC more than once in the file, number its rows and keep row 1.
# MAGIC Snowflake equivalent, which you'll know:
# MAGIC ```sql
# MAGIC SELECT * FROM labelled
# MAGIC QUALIFY ROW_NUMBER() OVER (PARTITION BY transaction_id ORDER BY <priority> DESC) = 1
# MAGIC ```
# MAGIC Spark has no `QUALIFY`, so we add the row number as a column, filter on it, then drop it.
# MAGIC
# MAGIC **Which row wins?** If the same transaction appears as both a change and a delete in one
# MAGIC file, the delete wins: **D beats C beats A**. Rows that are exact duplicates are identical,
# MAGIC so it doesn't matter which one is kept.
# MAGIC
# MAGIC Duplicates across *different* files aren't a problem: that's what the MERGE in step 7 handles.

# COMMAND ----------

from pyspark.sql import Window

status_priority = (
    F.when(F.col("record_status") == "D", 3)
     .when(F.col("record_status") == "C", 2)
     .otherwise(1)
)

latest_first = Window.partitionBy("transaction_id").orderBy(status_priority.desc())

changes_df = (
    labelled_df
    .withColumn("row_num", F.row_number().over(latest_first))
    .filter(F.col("row_num") == 1)
    .drop("row_num")
)

changes_row_count = changes_df.count()
duplicates_removed = bronze_row_count - changes_row_count
print(f"Bronze rows:          {bronze_row_count:,}")
print(f"Duplicates removed:   {duplicates_removed:,}")
print(f"Distinct changes:     {changes_row_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: MERGE into silver
# MAGIC
# MAGIC First a final `select` puts the columns in the same order as the silver table, adds
# MAGIC `silver_updated_at`, and drops the temporary `price_raw` / `date_raw` columns.
# MAGIC
# MAGIC **New concept: temp view.** `createOrReplaceTempView("silver_changes")` gives the DataFrame a
# MAGIC name that SQL can use, like a CTE that lasts for this session. That lets us write the MERGE in
# MAGIC plain SQL.
# MAGIC
# MAGIC The silver table is created on the first run. `NOT NULL` on `transaction_id` is enforced by
# MAGIC Delta, so a write with a NULL id fails instead of landing.
# MAGIC
# MAGIC The MERGE rules:
# MAGIC - Already in silver and the file says **D**: **delete** it
# MAGIC - Already in silver otherwise (**C**, or **A** again on a re-run): **update** the whole row
# MAGIC - Not in silver and **not D**: **insert** it
# MAGIC - Not in silver and **D**: nothing to delete, so ignore it (counted in step 8)
# MAGIC
# MAGIC `UPDATE SET *` and `INSERT *` mean "every column, matched by name". That's why the
# MAGIC column names have to match the table exactly.
# MAGIC
# MAGIC `record_status` is kept in silver as the *last change applied* to that row (A or C).

# COMMAND ----------

silver_columns = [
    "transaction_id", "price", "date_of_transfer",
    "postcode", "postcode_district", "postcode_valid",
    "property_type", "property_type_label",
    "old_new", "is_new_build",
    "duration", "duration_label",
    "paon", "saon", "street", "locality", "town_city", "district", "county",
    "ppd_category", "is_additional_ppd",
    "record_status", "source_file", "bronze_load_timestamp",
]

final_changes_df = (
    changes_df
    .select(*silver_columns)
    .withColumn("silver_updated_at", F.current_timestamp())
)
final_changes_df.createOrReplaceTempView("silver_changes")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {SILVER_TABLE} (
    transaction_id        STRING NOT NULL,
    price                 BIGINT,
    date_of_transfer      DATE,
    postcode              STRING,
    postcode_district     STRING,
    postcode_valid        BOOLEAN,
    property_type         STRING,
    property_type_label   STRING,
    old_new               STRING,
    is_new_build          BOOLEAN,
    duration              STRING,
    duration_label        STRING,
    paon                  STRING,
    saon                  STRING,
    street                STRING,
    locality              STRING,
    town_city             STRING,
    district              STRING,
    county                STRING,
    ppd_category          STRING,
    is_additional_ppd     BOOLEAN,
    record_status         STRING,
    source_file           STRING,
    bronze_load_timestamp TIMESTAMP,
    silver_updated_at     TIMESTAMP
)
COMMENT 'Price Paid Data, one row per transaction_id: latest version after applying A/C/D changes. Typed, postcode standardised, category B flagged (not removed).'
""")

silver_before = spark.table(SILVER_TABLE).count()

merge_result = spark.sql(f"""
MERGE INTO {SILVER_TABLE} AS t
USING silver_changes AS s
ON t.transaction_id = s.transaction_id
WHEN MATCHED AND s.record_status = 'D' THEN DELETE
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED AND s.record_status <> 'D' THEN INSERT *
""")

# Databricks returns one row of counts from a MERGE.
display(merge_result)
m = merge_result.first()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Reconcile the row counts
# MAGIC
# MAGIC Three checks, so every row is accounted for:
# MAGIC 1. **Bronze → changes:** bronze rows − duplicates removed = distinct changes
# MAGIC 2. **Changes → MERGE:** every non-D change was either inserted or updated
# MAGIC    (D rows were either deleted, or ignored because the sale wasn't in silver)
# MAGIC 3. **Silver:** rows before + inserted − deleted = rows after
# MAGIC
# MAGIC For the first run on `pp-2025.csv`, expect: silver before = 0, inserted = distinct changes,
# MAGIC updated = deleted = 0.

# COMMAND ----------

inserted = m["num_inserted_rows"]
updated = m["num_updated_rows"]
deleted = m["num_deleted_rows"]

delete_rows_in_file = final_changes_df.filter(F.col("record_status") == "D").count()
non_delete_rows_in_file = changes_row_count - delete_rows_in_file
ignored_deletes = delete_rows_in_file - deleted

silver_after = spark.table(SILVER_TABLE).count()

print(f"Bronze rows for file:            {bronze_row_count:,}")
print(f"  - duplicates removed:          {duplicates_removed:,}")
print(f"  = distinct changes:            {changes_row_count:,}")
print(f"      A/C rows:                  {non_delete_rows_in_file:,}  -> inserted {inserted:,} + updated {updated:,}")
print(f"      D rows:                    {delete_rows_in_file:,}  -> deleted {deleted:,} + not in silver {ignored_deletes:,}")
print()
print(f"Silver before:                   {silver_before:,}")
print(f"  + inserted - deleted:          {inserted - deleted:+,}")
print(f"  = silver after:                {silver_after:,}")

recon_problems = []
if bronze_row_count - duplicates_removed != changes_row_count:
    recon_problems.append("bronze minus duplicates does not equal distinct changes")
if inserted + updated != non_delete_rows_in_file:
    recon_problems.append(f"A/C rows ({non_delete_rows_in_file:,}) != inserted + updated ({inserted + updated:,})")
if silver_before + inserted - deleted != silver_after:
    recon_problems.append(f"silver before + inserted - deleted ({silver_before + inserted - deleted:,}) != silver after ({silver_after:,})")

if recon_problems:
    raise Exception("Reconciliation FAILED: " + "; ".join(recon_problems))

print("\nReconciliation passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Quality checks on the whole silver table
# MAGIC
# MAGIC **Fail** if `transaction_id` isn't unique (the MERGE should guarantee it).
# MAGIC **Report only:** invalid and missing postcodes, category B rows, and suspicious prices.
# MAGIC These are facts about the source data, not pipeline errors, and gold decides how to treat them.

# COMMAND ----------

silver_checks = spark.table(SILVER_TABLE).agg(
    F.count("*").alias("row_count"),
    F.countDistinct("transaction_id").alias("distinct_transaction_ids"),
    count_where(F.col("postcode").isNull()).alias("missing_postcodes"),
    count_where(F.col("postcode").isNotNull() & ~F.col("postcode_valid")).alias("invalid_postcodes"),
    count_where(F.col("is_additional_ppd")).alias("category_b_rows"),
    count_where(F.col("price") < 1000).alias("prices_under_1000"),
    F.countDistinct("postcode_district").alias("postcode_districts"),
    F.min("date_of_transfer").alias("min_date"),
    F.max("date_of_transfer").alias("max_date"),
)
display(silver_checks)

s = silver_checks.first()
if s["row_count"] != s["distinct_transaction_ids"]:
    raise Exception(f"Silver has duplicate transaction_ids: {s['row_count'] - s['distinct_transaction_ids']:,}")

print("Silver checks passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Have a look (SQL)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sales by property type, standard (A) vs additional (B)
# MAGIC SELECT property_type_label,
# MAGIC        is_additional_ppd,
# MAGIC        COUNT(*)                 AS sales,
# MAGIC        ROUND(AVG(price))        AS avg_price,
# MAGIC        PERCENTILE(price, 0.5)   AS median_price
# MAGIC FROM workspace.default.silver_price_paid
# MAGIC GROUP BY property_type_label, is_additional_ppd
# MAGIC ORDER BY property_type_label, is_additional_ppd

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Postcodes that didn't pass validation (to see what the source data looks like)
# MAGIC SELECT postcode, COUNT(*) AS sales
# MAGIC FROM workspace.default.silver_price_paid
# MAGIC WHERE postcode IS NOT NULL AND NOT postcode_valid
# MAGIC GROUP BY postcode
# MAGIC ORDER BY sales DESC
# MAGIC LIMIT 20
