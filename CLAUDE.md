# soldprices-pipeline

Learning and portfolio project: rebuild the soldprices.uk data pipeline as a proper
data engineering pipeline on Databricks, then extend it to Azure and add one
Azure AI Foundry feature.

## About me (how to work with me)
- I'm an experienced London Market data analyst/BA: strong SQL, Snowflake, dbt.
- I'm learning Python/PySpark. I want to be able to read and modify every line.
- Explain each new PySpark concept briefly the first time it appears, and relate it
  to the SQL equivalent where possible.
- Prefer small, readable steps over clever code. No unnecessary abstractions.
- Don't make changes to Databricks or Azure resources without telling me first.

## Data
- Source: HM Land Registry Price Paid Data (CSV, no header row).
- Columns in order: transaction_id, price, date_of_transfer, postcode, property_type,
  old_new, duration, paon, saon, street, locality, town_city, district, county,
  ppd_category, record_status
- record_status: A = add, C = change, D = delete (monthly update files).
- The monthly update file always has the same name (pp-monthly-update-new-version.csv).
  Land each one with a dated name (e.g. pp-monthly-2026-10.csv): bronze replaces rows
  by source_file, and silver must apply monthly files in date order.
- Start with one year (~1m rows); Databricks Free Edition compute is small.

## Platform
- Databricks Free Edition (serverless only, no R/Scala, non-commercial use).
- Catalog: workspace, schema: default (later: separate bronze/silver/gold schemas).
- Raw files land in the Volume: /Volumes/workspace/default/landing/

## Architecture (medallion)
- Bronze: raw files loaded with an explicit schema + load metadata
  (source file name, load timestamp).
- Silver: cleaned, typed, standardised (postcode, postcode_district, property type
  labels), deduplicated, record_status changes/deletes applied.
- Gold: aggregates for the site - by postcode district / town, property type, month.
- Every layer: row-count reconciliation and basic data quality checks.

## Phases
1. Pipeline on Databricks Free Edition, scheduled job, code in GitHub.
2. Azure: land monthly Land Registry file in Azure Storage automatically
   (Data Factory or Function). Spending limit set on the account.
3. Azure AI Foundry: natural-language search over gold tables (question -> SQL).
4. Write-up and architecture diagram for jcreaney.com.

## Conventions
- Notebooks/code in /notebooks, SQL in /sql, docs in /docs.
- Each notebook starts with a markdown cell explaining what it does and why.
- Commit small, with clear messages.

## Public repo rules
- This repo is PUBLIC. Never commit credentials: Databricks tokens, .databrickscfg,
  .env files, Azure keys or connection strings. Use Databricks secrets / env vars.
- Keep anything specific to the live soldprices.uk site out of this repo entirely:
  no database connection details, server names, resource names or file paths.
  This project only uses public Land Registry data and its own Databricks/Azure resources.
