# Databricks notebook source
# MAGIC %md
# MAGIC # GMR Mosaic AI Demo - Data Ingestion Patterns
# MAGIC
# MAGIC **Business Context:** As a data engineer at GMR, you need to continuously ingest performance logs
# MAGIC from various partners. This notebook demonstrates the Auto Loader patterns that would be used
# MAGIC in production for incremental CSV ingestion with schema evolution.
# MAGIC
# MAGIC ## Patterns Demonstrated
# MAGIC - **Auto Loader (Structured Streaming)** for incremental CSV ingestion
# MAGIC - **Schema Evolution** to handle new columns without breaking pipelines
# MAGIC - **Data Quality Checks** using expectations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "gmr_demo_catalog", "Catalog Name")
dbutils.widgets.text("schema", "royalties", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto Loader Pattern (Reference)
# MAGIC
# MAGIC In production, you would use Auto Loader to ingest CSV files from cloud storage:
# MAGIC
# MAGIC ```python
# MAGIC # Auto Loader configuration for CSV ingestion
# MAGIC raw_performance_stream = (
# MAGIC     spark.readStream
# MAGIC     .format("cloudFiles")
# MAGIC     .option("cloudFiles.format", "csv")
# MAGIC     .option("cloudFiles.schemaLocation", "/path/to/schema")
# MAGIC     .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
# MAGIC     .option("cloudFiles.inferColumnTypes", "true")
# MAGIC     .option("header", "true")
# MAGIC     .load("/Volumes/catalog/schema/raw_data/")
# MAGIC     .withColumn("_ingested_at", current_timestamp())
# MAGIC     .withColumn("_source_file", input_file_name())
# MAGIC )
# MAGIC
# MAGIC # Write to Delta with schema evolution
# MAGIC query = (
# MAGIC     raw_performance_stream.writeStream
# MAGIC     .format("delta")
# MAGIC     .outputMode("append")
# MAGIC     .option("mergeSchema", "true")
# MAGIC     .trigger(availableNow=True)
# MAGIC     .toTable("performance_logs_streaming")
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Existing Data
# MAGIC
# MAGIC For this demo, we'll verify the data created by the data generation notebook.

# COMMAND ----------

from pyspark.sql import functions as F

# Check performance_logs table from data generation
performance_logs = spark.table(f"{CATALOG}.{SCHEMA}.performance_logs")
print(f"Performance logs count: {performance_logs.count()}")

# COMMAND ----------

# Show sample data
display(performance_logs.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Validation
# MAGIC
# MAGIC Apply data quality expectations to ensure data meets business rules.

# COMMAND ----------

# Reference data for validation
TERRITORIES = ["US", "CA", "UK", "DE", "FR", "JP", "AU", "BR", "MX", "KR", "IN", "ES", "IT", "NL", "SE"]

# Define quality checks
quality_df = (
    performance_logs
    .withColumn("is_valid_song_id", F.col("song_id").rlike("^SONG[0-9]{6}$"))
    .withColumn("is_valid_duration", (F.col("duration_played") > 0) & (F.col("duration_played") < 3600))
    .withColumn("is_valid_territory", F.col("territory").isin(*TERRITORIES))
)

# Summarize quality
quality_summary = quality_df.agg(
    F.count("*").alias("total_records"),
    F.sum(F.when(F.col("is_valid_song_id"), 1).otherwise(0)).alias("valid_song_ids"),
    F.sum(F.when(F.col("is_valid_duration"), 1).otherwise(0)).alias("valid_durations"),
    F.sum(F.when(F.col("is_valid_territory"), 1).otherwise(0)).alias("valid_territories")
)

display(quality_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestion Metrics
# MAGIC
# MAGIC Track performance log volume by platform and territory.

# COMMAND ----------

# Volume by platform
platform_metrics = (
    performance_logs
    .groupBy("platform")
    .agg(
        F.count("*").alias("play_count"),
        F.avg("duration_played").alias("avg_duration")
    )
    .orderBy("play_count", ascending=False)
)

display(platform_metrics)

# COMMAND ----------

# Volume by territory
territory_metrics = (
    performance_logs
    .groupBy("territory")
    .agg(
        F.count("*").alias("play_count"),
        F.countDistinct("song_id").alias("unique_songs")
    )
    .orderBy("play_count", ascending=False)
)

display(territory_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC 1. **Auto Loader pattern** for incremental CSV ingestion (reference code)
# MAGIC 2. **Schema evolution** configuration options
# MAGIC 3. **Data quality validation** on ingested records
# MAGIC 4. **Volume metrics** by platform and territory
# MAGIC
# MAGIC ### Next Steps
# MAGIC Proceed to **03_feature_engineering.py** to create ML-ready features from the data.

# COMMAND ----------

print(f"""
Data Ingestion Patterns Complete!
=================================
Catalog: {CATALOG}
Schema: {SCHEMA}

Performance Logs: {performance_logs.count()} records
Platforms: {performance_logs.select('platform').distinct().count()} unique
Territories: {performance_logs.select('territory').distinct().count()} unique
""")
