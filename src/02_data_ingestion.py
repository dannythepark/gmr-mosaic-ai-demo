# Databricks notebook source
# MAGIC %md
# MAGIC # GMR Mosaic AI Demo - Data Ingestion
# MAGIC
# MAGIC **Business Context:** As a data engineer at GMR, you need to continuously ingest performance logs
# MAGIC from various partners (streaming services, radio stations, venues). This data arrives as CSV files
# MAGIC dropped into cloud storage and must be processed incrementally with schema evolution support.
# MAGIC
# MAGIC ## Patterns Demonstrated
# MAGIC - **Auto Loader (Structured Streaming)** for incremental CSV ingestion
# MAGIC - **Schema Evolution** to handle new columns without breaking pipelines
# MAGIC - **Data Quality Checks** using expectations
# MAGIC - **Volume-based storage** for raw file landing zone

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "gmr_demo", "Catalog Name")
dbutils.widgets.text("schema", "royalties", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# Volume paths for raw data landing zone
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_data"
CHECKPOINT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/performance_logs"

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Landing Zone: {VOLUME_PATH}")

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Volume for Raw Data Landing Zone

# COMMAND ----------

# Create volume for raw data if it doesn't exist
spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_data
    COMMENT 'Landing zone for raw performance log CSV files from partners'
""")

spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.checkpoints
    COMMENT 'Checkpoint storage for streaming jobs'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Sample CSV Files for Demo
# MAGIC
# MAGIC In production, these CSVs would arrive from partner systems. For the demo,
# MAGIC we'll generate sample files to simulate the incoming data flow.

# COMMAND ----------

import pandas as pd
from datetime import datetime, timedelta
import random
import os

# Reference data matching our data model
PLATFORMS = [
    "Spotify", "Apple Music", "Amazon Music", "YouTube Music", "Pandora",
    "KROQ-FM", "Z100", "BBC Radio 1", "Madison Square Garden", "The Forum"
]
TERRITORIES = ["US", "CA", "UK", "DE", "FR", "JP", "AU", "BR", "MX", "KR"]
REPORTERS = ["PartnerSync", "DirectFeed", "ManualUpload", "APIIngestion"]

# Get existing song IDs
songs_df = spark.table(f"{CATALOG}.{SCHEMA}.songs")
song_ids = [row.song_id for row in songs_df.select("song_id").limit(100).collect()]

def generate_performance_csv(num_records, batch_id, include_new_column=False):
    """Generate a sample performance log CSV file."""
    records = []
    for i in range(num_records):
        record = {
            "log_id": f"BATCH{batch_id}_LOG{i:06d}",
            "song_id": random.choice(song_ids),
            "platform": random.choice(PLATFORMS),
            "play_timestamp": (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
            "territory": random.choice(TERRITORIES),
            "duration_played": random.randint(30, 300),
            "reported_by": random.choice(REPORTERS)
        }
        # Simulate schema evolution - new column appearing in later batches
        if include_new_column:
            record["listener_type"] = random.choice(["premium", "free", "trial"])
        records.append(record)
    return pd.DataFrame(records)

# Generate sample CSV files
csv_dir = f"/dbfs{VOLUME_PATH}/performance_logs"
os.makedirs(csv_dir, exist_ok=True)

# Batch 1 - Original schema
df1 = generate_performance_csv(500, batch_id=1, include_new_column=False)
df1.to_csv(f"{csv_dir}/performance_batch_001.csv", index=False)

# Batch 2 - Original schema
df2 = generate_performance_csv(500, batch_id=2, include_new_column=False)
df2.to_csv(f"{csv_dir}/performance_batch_002.csv", index=False)

# Batch 3 - Schema evolution: new column added
df3 = generate_performance_csv(500, batch_id=3, include_new_column=True)
df3.to_csv(f"{csv_dir}/performance_batch_003.csv", index=False)

print(f"Generated 3 sample CSV batches in {VOLUME_PATH}/performance_logs/")

# COMMAND ----------

# Verify files were created
display(dbutils.fs.ls(f"{VOLUME_PATH}/performance_logs/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto Loader Ingestion Pattern
# MAGIC
# MAGIC Auto Loader provides:
# MAGIC - **Incremental processing** - only new files are processed
# MAGIC - **Schema inference** - automatically detects CSV schema
# MAGIC - **Schema evolution** - handles new columns gracefully
# MAGIC - **Exactly-once semantics** - through checkpoint tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Auto Loader Stream

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, input_file_name

# Auto Loader configuration for CSV ingestion
raw_performance_stream = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", f"{CHECKPOINT_PATH}/schema")
    # Enable schema evolution to handle new columns
    .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
    .option("cloudFiles.inferColumnTypes", "true")
    # CSV-specific options
    .option("header", "true")
    .option("multiLine", "false")
    .load(f"{VOLUME_PATH}/performance_logs/")
    # Add metadata columns
    .withColumn("_ingested_at", current_timestamp())
    .withColumn("_source_file", input_file_name())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to Delta Table with Schema Evolution

# COMMAND ----------

# Define the target table
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.performance_logs_streaming"

# Write stream to Delta table
query = (
    raw_performance_stream.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/delta")
    # Enable schema evolution on the Delta table
    .option("mergeSchema", "true")
    .trigger(availableNow=True)  # Process all available files then stop
    .toTable(TARGET_TABLE)
)

# Wait for the stream to complete
query.awaitTermination()

print(f"Streaming ingestion complete. Data written to {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Ingested Data

# COMMAND ----------

# Check the ingested data
ingested_df = spark.table(TARGET_TABLE)
print(f"Total records ingested: {ingested_df.count()}")
print(f"\nSchema (note the evolved 'listener_type' column):")
ingested_df.printSchema()

# COMMAND ----------

# Display sample records
display(ingested_df.orderBy("_ingested_at", ascending=False).limit(10))

# COMMAND ----------

# Show records with the new evolved column
display(
    ingested_df
    .filter("listener_type IS NOT NULL")
    .select("log_id", "song_id", "platform", "listener_type", "_source_file")
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakeflow Connect Pattern (Alternative)
# MAGIC
# MAGIC For enterprise-scale ingestion from SaaS applications, **Lakeflow Connect** provides
# MAGIC managed connectors. Here's how you would configure ingestion from a hypothetical
# MAGIC partner API:
# MAGIC
# MAGIC ```python
# MAGIC # Lakeflow Connect example (conceptual - requires connector setup)
# MAGIC from databricks.connect import ingestion
# MAGIC
# MAGIC # Configure the ingestion pipeline
# MAGIC pipeline = ingestion.create_pipeline(
# MAGIC     name="gmr_spotify_performance_feed",
# MAGIC     source={
# MAGIC         "type": "rest_api",
# MAGIC         "config": {
# MAGIC             "base_url": "https://api.partner.spotify.com/v1",
# MAGIC             "endpoint": "/performance-reports",
# MAGIC             "auth_type": "oauth2",
# MAGIC             "secret_scope": "gmr-secrets",
# MAGIC             "secret_key": "spotify-api-token"
# MAGIC         }
# MAGIC     },
# MAGIC     destination={
# MAGIC         "catalog": "gmr_demo",
# MAGIC         "schema": "royalties",
# MAGIC         "table": "spotify_performance_raw"
# MAGIC     },
# MAGIC     schedule="0 */6 * * *"  # Every 6 hours
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Validation
# MAGIC
# MAGIC Apply data quality expectations to ensure incoming data meets business rules.

# COMMAND ----------

from pyspark.sql import functions as F

# Define quality checks
quality_df = (
    spark.table(TARGET_TABLE)
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
# MAGIC ## Monitoring Ingestion Jobs
# MAGIC
# MAGIC In production, you would set up alerts for:
# MAGIC - **Ingestion lag** - files not processed within SLA
# MAGIC - **Schema drift** - unexpected new columns
# MAGIC - **Data quality failures** - records failing validation
# MAGIC - **Volume anomalies** - unusual spikes or drops in record counts

# COMMAND ----------

# Get ingestion metrics
ingestion_metrics = (
    spark.table(TARGET_TABLE)
    .groupBy(F.date_trunc("hour", "_ingested_at").alias("ingestion_hour"))
    .agg(
        F.count("*").alias("records_ingested"),
        F.countDistinct("_source_file").alias("files_processed")
    )
    .orderBy("ingestion_hour", ascending=False)
)

display(ingestion_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC 1. **Auto Loader** for incremental CSV ingestion from cloud storage
# MAGIC 2. **Schema Evolution** handling new columns without pipeline failures
# MAGIC 3. **Volume-based storage** as a managed landing zone
# MAGIC 4. **Data quality validation** for incoming records
# MAGIC
# MAGIC ### Next Steps
# MAGIC Proceed to **03_feature_engineering.py** to create ML-ready features from the ingested data.

# COMMAND ----------

print(f"""
Data Ingestion Complete!
========================
Source: {VOLUME_PATH}/performance_logs/
Target: {TARGET_TABLE}
Records Ingested: {ingested_df.count()}

Schema Evolution: Successfully handled new 'listener_type' column
""")
