# Databricks notebook source
# MAGIC %md
# MAGIC # GMR Mosaic AI Demo - Feature Engineering
# MAGIC
# MAGIC **Business Context:** As a data scientist at GMR, you need ML-ready features to power
# MAGIC predictive models and analytics dashboards. This notebook creates a Feature Store table
# MAGIC with computed features that capture song popularity, royalty efficiency, geographic
# MAGIC distribution, and licensee reliability.
# MAGIC
# MAGIC ## Features Created
# MAGIC | Feature | Description | Business Value |
# MAGIC |---------|-------------|----------------|
# MAGIC | `song_popularity_score` | Rolling 90-day play count, normalized 0-100 | Identify trending songs for licensing deals |
# MAGIC | `royalty_per_play` | Average net royalty per performance | Track monetization efficiency |
# MAGIC | `territory_concentration` | Herfindahl index of play distribution | Assess geographic diversification |
# MAGIC | `licensee_risk_score` | Late/missing payment ratio per licensee | Flag high-risk partners |

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
# MAGIC ## Load Source Tables

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta

# Load all source tables
songs_df = spark.table(f"{CATALOG}.{SCHEMA}.songs")
performance_logs_df = spark.table(f"{CATALOG}.{SCHEMA}.performance_logs")
royalty_payments_df = spark.table(f"{CATALOG}.{SCHEMA}.royalty_payments")
licenses_df = spark.table(f"{CATALOG}.{SCHEMA}.licenses")
songwriters_df = spark.table(f"{CATALOG}.{SCHEMA}.songwriters")

print(f"Songs: {songs_df.count()} records")
print(f"Performance Logs: {performance_logs_df.count()} records")
print(f"Royalty Payments: {royalty_payments_df.count()} records")
print(f"Licenses: {licenses_df.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature 1: Song Popularity Score
# MAGIC
# MAGIC Calculate a rolling 90-day play count for each song, normalized to a 0-100 scale.
# MAGIC This helps identify trending songs that should be prioritized for new licensing deals.

# COMMAND ----------

# Calculate the date 90 days ago
cutoff_date = datetime.now() - timedelta(days=90)

# Count plays per song in the last 90 days
popularity_df = (
    performance_logs_df
    .filter(F.col("play_timestamp") >= cutoff_date)
    .groupBy("song_id")
    .agg(F.count("*").alias("play_count_90d"))
)

# Get min/max for normalization
stats = popularity_df.agg(
    F.min("play_count_90d").alias("min_plays"),
    F.max("play_count_90d").alias("max_plays")
).collect()[0]

min_plays = stats["min_plays"]
max_plays = stats["max_plays"]

# Normalize to 0-100 scale
popularity_df = popularity_df.withColumn(
    "song_popularity_score",
    F.round(
        (F.col("play_count_90d") - min_plays) / (max_plays - min_plays) * 100,
        2
    )
)

# Handle songs with no recent plays
popularity_df = (
    songs_df.select("song_id")
    .join(popularity_df, "song_id", "left")
    .fillna({"play_count_90d": 0, "song_popularity_score": 0.0})
)

display(popularity_df.orderBy("song_popularity_score", ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature 2: Royalty Per Play
# MAGIC
# MAGIC Calculate the average net royalty earned per performance for each song.
# MAGIC This metric helps identify which songs generate the most revenue per play.

# COMMAND ----------

# Calculate total performances per song
plays_per_song = (
    performance_logs_df
    .groupBy("song_id")
    .agg(F.count("*").alias("total_plays"))
)

# Calculate total net royalties per song
royalties_per_song = (
    royalty_payments_df
    .groupBy("song_id")
    .agg(F.sum("net_amount").alias("total_net_royalties"))
)

# Join and calculate royalty per play
royalty_per_play_df = (
    plays_per_song
    .join(royalties_per_song, "song_id", "left")
    .fillna({"total_net_royalties": 0})
    .withColumn(
        "royalty_per_play",
        F.when(F.col("total_plays") > 0,
               F.round(F.col("total_net_royalties") / F.col("total_plays"), 4))
        .otherwise(0.0)
    )
    .select("song_id", "total_plays", "total_net_royalties", "royalty_per_play")
)

display(royalty_per_play_df.orderBy("royalty_per_play", ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature 3: Territory Concentration (Herfindahl Index)
# MAGIC
# MAGIC The Herfindahl-Hirschman Index (HHI) measures market concentration. For GMR, we use it
# MAGIC to measure how concentrated a song's plays are across territories:
# MAGIC - **Low HHI (closer to 0)**: Plays distributed across many territories (diversified)
# MAGIC - **High HHI (closer to 1)**: Plays concentrated in few territories (concentrated)
# MAGIC
# MAGIC This helps identify songs with global appeal vs. regional hits.

# COMMAND ----------

# Calculate play distribution by territory for each song
territory_plays = (
    performance_logs_df
    .groupBy("song_id", "territory")
    .agg(F.count("*").alias("territory_plays"))
)

# Calculate total plays per song
total_plays_window = Window.partitionBy("song_id")
territory_shares = territory_plays.withColumn(
    "total_plays",
    F.sum("territory_plays").over(total_plays_window)
).withColumn(
    "market_share",
    F.col("territory_plays") / F.col("total_plays")
).withColumn(
    "share_squared",
    F.pow(F.col("market_share"), 2)
)

# Calculate HHI (sum of squared market shares)
hhi_df = (
    territory_shares
    .groupBy("song_id")
    .agg(
        F.round(F.sum("share_squared"), 4).alias("territory_concentration"),
        F.countDistinct("territory").alias("num_territories")
    )
)

# Ensure all songs are included
territory_concentration_df = (
    songs_df.select("song_id")
    .join(hhi_df, "song_id", "left")
    .fillna({"territory_concentration": 1.0, "num_territories": 0})  # No plays = max concentration
)

display(territory_concentration_df.orderBy("territory_concentration").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature 4: Licensee Risk Score
# MAGIC
# MAGIC Calculate a risk score for each licensee based on their payment history.
# MAGIC A high risk score indicates a partner with frequent late, disputed, or missing payments.

# COMMAND ----------

# Join payments with licenses to get licensee information
payments_with_licensee = (
    royalty_payments_df
    .join(licenses_df.select("song_id", "licensee_name").distinct(), "song_id", "left")
)

# Calculate payment reliability metrics per licensee
licensee_risk_df = (
    payments_with_licensee
    .groupBy("licensee_name")
    .agg(
        F.count("*").alias("total_payments"),
        F.sum(F.when(F.col("payment_status") == "completed", 1).otherwise(0)).alias("completed_payments"),
        F.sum(F.when(F.col("payment_status") == "disputed", 1).otherwise(0)).alias("disputed_payments"),
        F.sum(F.when(F.col("payment_status") == "on_hold", 1).otherwise(0)).alias("on_hold_payments"),
        F.sum(F.when(F.col("payment_status").isin("pending", "processing"), 1).otherwise(0)).alias("pending_payments")
    )
    .withColumn(
        "licensee_risk_score",
        F.round(
            (F.col("disputed_payments") + F.col("on_hold_payments") + F.col("pending_payments") * 0.5)
            / F.col("total_payments"),
            4
        )
    )
    .fillna({"licensee_risk_score": 0.0})
)

display(licensee_risk_df.orderBy("licensee_risk_score", ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Store Table
# MAGIC
# MAGIC Combine all features into a single Feature Store table in Unity Catalog.
# MAGIC The table will be keyed by `song_id` for easy lookup and joining.

# COMMAND ----------

# Combine all features into a single DataFrame
song_features_df = (
    songs_df.select("song_id", "title", "artist_name", "genre")
    # Add popularity score
    .join(
        popularity_df.select("song_id", "play_count_90d", "song_popularity_score"),
        "song_id",
        "left"
    )
    # Add royalty per play
    .join(
        royalty_per_play_df.select("song_id", "royalty_per_play", "total_plays", "total_net_royalties"),
        "song_id",
        "left"
    )
    # Add territory concentration
    .join(
        territory_concentration_df.select("song_id", "territory_concentration", "num_territories"),
        "song_id",
        "left"
    )
    # Add feature computation timestamp
    .withColumn("feature_timestamp", F.current_timestamp())
    # Fill nulls with defaults
    .fillna({
        "play_count_90d": 0,
        "song_popularity_score": 0.0,
        "royalty_per_play": 0.0,
        "total_plays": 0,
        "total_net_royalties": 0.0,
        "territory_concentration": 1.0,
        "num_territories": 0
    })
)

# COMMAND ----------

# Display the combined features
display(song_features_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save as Delta Table

# COMMAND ----------

# Save song features as Delta table
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.song_features")

song_features_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.song_features")

print(f"Feature table created: {CATALOG}.{SCHEMA}.song_features")

# COMMAND ----------

# Verify the feature table
display(spark.table(f"{CATALOG}.{SCHEMA}.song_features").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Licensee Risk Feature Table
# MAGIC
# MAGIC Create a separate feature table for licensee-level risk scores.

# COMMAND ----------

# Add timestamp column
licensee_features_df = licensee_risk_df.withColumn("feature_timestamp", F.current_timestamp())

# Save licensee features as Delta table
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.licensee_features")

licensee_features_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.licensee_features")

print(f"Feature table created: {CATALOG}.{SCHEMA}.licensee_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Summary Statistics

# COMMAND ----------

# Summary statistics for song features
song_features_summary = (
    spark.table(f"{CATALOG}.{SCHEMA}.song_features")
    .select(
        "song_popularity_score",
        "royalty_per_play",
        "territory_concentration"
    )
    .summary()
)

display(song_features_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Distribution Analysis

# COMMAND ----------

import matplotlib.pyplot as plt

# Get features as pandas for plotting
features_pd = spark.table(f"{CATALOG}.{SCHEMA}.song_features").toPandas()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Popularity Score Distribution
axes[0].hist(features_pd["song_popularity_score"], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel("Popularity Score")
axes[0].set_ylabel("Count")
axes[0].set_title("Song Popularity Score Distribution")

# Royalty Per Play Distribution
axes[1].hist(features_pd["royalty_per_play"], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel("Royalty Per Play ($)")
axes[1].set_ylabel("Count")
axes[1].set_title("Royalty Per Play Distribution")

# Territory Concentration Distribution
axes[2].hist(features_pd["territory_concentration"], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[2].set_xlabel("HHI (Territory Concentration)")
axes[2].set_ylabel("Count")
axes[2].set_title("Territory Concentration Distribution")

plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog Lineage
# MAGIC
# MAGIC The feature tables now have full lineage tracked in Unity Catalog:
# MAGIC - `songs` → `song_features`
# MAGIC - `performance_logs` → `song_features`
# MAGIC - `royalty_payments` → `song_features`, `licensee_features`
# MAGIC - `licenses` → `licensee_features`
# MAGIC
# MAGIC Navigate to the Unity Catalog explorer to view the lineage graph.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook created:
# MAGIC 1. **`song_features`** - Feature table with popularity, royalty efficiency, and territory metrics
# MAGIC 2. **`licensee_features`** - Feature table with licensee risk scores
# MAGIC
# MAGIC These features are now available for:
# MAGIC - ML model training (churn prediction, revenue forecasting)
# MAGIC - Real-time feature lookup via Model Serving
# MAGIC - Analytics dashboards
# MAGIC
# MAGIC ### Next Steps
# MAGIC Proceed to **04_vector_index.py** to create the vector search index for semantic song discovery.

# COMMAND ----------

print(f"""
Feature Engineering Complete!
=============================
Feature Tables Created:
- {CATALOG}.{SCHEMA}.song_features ({song_features_df.count()} songs)
- {CATALOG}.{SCHEMA}.licensee_features ({licensee_features_df.count()} licensees)

Features:
- song_popularity_score: Rolling 90-day normalized play count
- royalty_per_play: Average net royalty per performance
- territory_concentration: HHI of geographic distribution
- licensee_risk_score: Payment reliability metric
""")
