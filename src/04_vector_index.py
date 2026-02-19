# Databricks notebook source
# MAGIC %md
# MAGIC # GMR Mosaic AI Demo - Vector Search Index
# MAGIC
# MAGIC **Business Context:** As a royalty analyst at GMR, you want to discover songs using natural
# MAGIC language queries like "upbeat pop songs by female artists released in 2024" or "acoustic
# MAGIC ballads about heartbreak." This notebook creates a vector search index that powers semantic
# MAGIC song discovery in our Mosaic AI agent.
# MAGIC
# MAGIC ## What We'll Build
# MAGIC 1. **Embedding Source Table** - Combine song metadata into searchable text chunks
# MAGIC 2. **Embedding Generation** - Use `databricks-gte-large-en` model for embeddings
# MAGIC 3. **Vector Search Index** - Create a Delta Sync index for real-time search

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "gmr_demo_catalog", "Catalog Name")
dbutils.widgets.text("schema", "royalties", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# Vector search configuration
VS_ENDPOINT_NAME = "gmr-vector-search-endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.song_metadata_index"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.song_metadata_embeddings"
EMBEDDING_MODEL = "databricks-gte-large-en"
EMBEDDING_DIM = 1024

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Vector Search Endpoint: {VS_ENDPOINT_NAME}")
print(f"Index Name: {VS_INDEX_NAME}")

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Embedding Source Table
# MAGIC
# MAGIC Combine song metadata with songwriter information into rich text descriptions
# MAGIC that will be embedded for semantic search.

# COMMAND ----------

from pyspark.sql import functions as F

# Load source tables
songs_df = spark.table(f"{CATALOG}.{SCHEMA}.songs")
songwriters_df = spark.table(f"{CATALOG}.{SCHEMA}.songwriters")
features_df = spark.table(f"{CATALOG}.{SCHEMA}.song_features")

# Create songwriter lookup
songwriter_names = (
    songwriters_df
    .select("songwriter_id", "name", "pro_affiliation")
    .withColumnRenamed("name", "songwriter_name")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Rich Text Descriptions
# MAGIC
# MAGIC We combine multiple fields into a single searchable text that captures
# MAGIC the essence of each song for semantic matching.

# COMMAND ----------

# Explode songwriter IDs and join with names
songs_with_writers = (
    songs_df
    .withColumn("songwriter_id", F.explode(F.split("songwriters", ",")))
    .join(songwriter_names, "songwriter_id", "left")
    .groupBy("song_id", "title", "artist_name", "publisher", "genre", "release_date", "isrc_code", "duration_seconds")
    .agg(
        F.concat_ws(", ", F.collect_list("songwriter_name")).alias("songwriter_names"),
        F.concat_ws(", ", F.collect_set("pro_affiliation")).alias("pro_affiliations")
    )
)

# Join with features for additional context
songs_enriched = (
    songs_with_writers
    .join(
        features_df.select("song_id", "song_popularity_score", "num_territories"),
        "song_id",
        "left"
    )
)

# Create the metadata text for embedding
song_metadata_df = (
    songs_enriched
    .withColumn(
        "metadata_text",
        F.concat_ws(" | ",
            F.concat(F.lit("Title: "), F.col("title")),
            F.concat(F.lit("Artist: "), F.col("artist_name")),
            F.concat(F.lit("Genre: "), F.col("genre")),
            F.concat(F.lit("Songwriters: "), F.col("songwriter_names")),
            F.concat(F.lit("Publisher: "), F.col("publisher")),
            F.concat(F.lit("Released: "), F.col("release_date").cast("string")),
            F.concat(F.lit("ISRC: "), F.col("isrc_code")),
            F.concat(F.lit("Duration: "), (F.col("duration_seconds") / 60).cast("int"), F.lit(" minutes")),
            F.concat(F.lit("Popularity: "), F.col("song_popularity_score").cast("string"), F.lit("/100")),
            F.concat(F.lit("Territories: "), F.col("num_territories").cast("string"))
        )
    )
    .select(
        "song_id",
        "title",
        "artist_name",
        "genre",
        "songwriter_names",
        "publisher",
        "release_date",
        "isrc_code",
        "metadata_text"
    )
)

display(song_metadata_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Source Table for Vector Search
# MAGIC
# MAGIC With computed embeddings mode, we don't need to generate embeddings manually.
# MAGIC The Vector Search index will automatically use the Foundation Model endpoint
# MAGIC `databricks-gte-large-en` to embed the `metadata_text` column.

# COMMAND ----------

# Write source table with Change Data Feed enabled for Delta Sync
song_metadata_df.write.mode("overwrite").option("overwriteSchema", "true").option("delta.enableChangeDataFeed", "true").saveAsTable(SOURCE_TABLE)

# Ensure CDF is enabled (in case table already existed)
spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"Source table created: {SOURCE_TABLE}")
print(f"Row count: {spark.table(SOURCE_TABLE).count()}")

# COMMAND ----------

# Verify the source table
display(
    spark.table(SOURCE_TABLE)
    .select("song_id", "title", "genre", "metadata_text")
    .limit(5)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Endpoint
# MAGIC
# MAGIC A Vector Search endpoint is a managed compute resource that serves vector search queries.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Initialize the Vector Search client
vsc = VectorSearchClient()

# COMMAND ----------

# Create the endpoint if it doesn't exist
try:
    endpoint = vsc.get_endpoint(VS_ENDPOINT_NAME)
    print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists")
except:
    print(f"Creating endpoint '{VS_ENDPOINT_NAME}'...")
    vsc.create_endpoint(
        name=VS_ENDPOINT_NAME,
        endpoint_type="STANDARD"
    )
    print(f"Endpoint '{VS_ENDPOINT_NAME}' created successfully")

# COMMAND ----------

# Wait for endpoint to be ready
import time

def wait_for_endpoint(endpoint_name, timeout=600):
    """Wait for endpoint to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            endpoint = vsc.get_endpoint(endpoint_name)
            # Handle both dict and object return types
            if isinstance(endpoint, dict):
                status = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")
            else:
                endpoint_status = getattr(endpoint, 'endpoint_status', None)
                if endpoint_status:
                    status = getattr(endpoint_status, 'state', 'UNKNOWN')
                else:
                    status = "UNKNOWN"
            if status == "ONLINE":
                print(f"Endpoint '{endpoint_name}' is ONLINE")
                return True
            print(f"Endpoint status: {status}. Waiting...")
        except Exception as e:
            print(f"Error checking endpoint status: {e}")
        time.sleep(30)
    raise TimeoutError(f"Endpoint '{endpoint_name}' did not become ready within {timeout} seconds")

wait_for_endpoint(VS_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Index
# MAGIC
# MAGIC Create a Delta Sync index that automatically stays in sync with the source table.

# COMMAND ----------

# Check if index already exists and delete if so (for idempotent execution)
try:
    existing_index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    print(f"Index '{VS_INDEX_NAME}' already exists. Deleting...")
    vsc.delete_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    time.sleep(30)  # Wait for deletion
except:
    print(f"Index '{VS_INDEX_NAME}' does not exist. Creating new index...")

# COMMAND ----------

# Create the Delta Sync index with computed embeddings
# This allows VECTOR_SEARCH SQL function to auto-embed text queries
index = vsc.create_delta_sync_index(
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME,
    source_table_name=SOURCE_TABLE,
    pipeline_type="TRIGGERED",  # Manual sync trigger
    primary_key="song_id",
    embedding_source_column="metadata_text",  # Text column to embed
    embedding_model_endpoint_name=EMBEDDING_MODEL,  # Foundation Model for embeddings
    # Columns to include in search results
    columns_to_sync=["song_id", "title", "artist_name", "genre", "songwriter_names", "publisher", "metadata_text"]
)

print(f"Vector Search index created: {VS_INDEX_NAME}")

# COMMAND ----------

# Wait for index to be ready
def wait_for_index(endpoint_name, index_name, timeout=1200):
    """Wait for index to be ready."""
    start_time = time.time()
    first_check = True
    while time.time() - start_time < timeout:
        try:
            index = vsc.get_index(endpoint_name, index_name)

            # Debug: print available attributes on first check
            if first_check:
                print(f"Index object type: {type(index)}")
                print(f"Index attributes: {[a for a in dir(index) if not a.startswith('_')]}")
                first_check = False

            # Try multiple ways to get status
            status = None
            detailed_state = "UNKNOWN"

            # Method 1: Direct attribute access
            if hasattr(index, 'status'):
                status = index.status
                if hasattr(status, 'detailed_state'):
                    detailed_state = status.detailed_state
                elif hasattr(status, 'ready'):
                    detailed_state = "ONLINE" if status.ready else "PROVISIONING"

            # Method 2: Try index_status attribute
            elif hasattr(index, 'index_status'):
                status = index.index_status
                if isinstance(status, str):
                    detailed_state = status
                elif hasattr(status, 'detailed_state'):
                    detailed_state = status.detailed_state

            # Method 3: describe() method
            elif hasattr(index, 'describe'):
                desc = index.describe()
                if isinstance(desc, dict):
                    detailed_state = desc.get('status', {}).get('detailed_state', 'UNKNOWN')

            # Accept any ONLINE state (ONLINE, ONLINE_NO_PENDING_UPDATE, etc.)
            if detailed_state.startswith("ONLINE"):
                print(f"Index '{index_name}' is {detailed_state}")
                return True
            print(f"Index status: {detailed_state}. Waiting...")
        except Exception as e:
            print(f"Error checking index status: {e}")
        time.sleep(30)
    raise TimeoutError(f"Index '{index_name}' did not become ready within {timeout} seconds")

wait_for_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Vector Search
# MAGIC
# MAGIC Let's test the semantic search capability with some sample queries.

# COMMAND ----------

def search_songs(query_text, num_results=5):
    """Search for songs using semantic similarity."""
    # With computed embeddings, we pass query_text directly
    # The index will automatically embed the query using the configured model
    index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    results = index.similarity_search(
        query_text=query_text,
        num_results=num_results,
        columns=["song_id", "title", "artist_name", "genre", "songwriter_names"]
    )

    return results

# COMMAND ----------

# Test search: Find upbeat pop songs
print("Query: 'upbeat pop songs by female artists'")
print("-" * 50)
results = search_songs("upbeat pop songs by female artists")
for row in results.get("result", {}).get("data_array", []):
    print(f"  {row[1]} by {row[2]} ({row[3]})")

# COMMAND ----------

# Test search: Find acoustic ballads
print("\nQuery: 'acoustic ballad about heartbreak'")
print("-" * 50)
results = search_songs("acoustic ballad about heartbreak")
for row in results.get("result", {}).get("data_array", []):
    print(f"  {row[1]} by {row[2]} ({row[3]})")

# COMMAND ----------

# Test search: Find electronic dance music
print("\nQuery: 'electronic dance music with heavy bass'")
print("-" * 50)
results = search_songs("electronic dance music with heavy bass")
for row in results.get("result", {}).get("data_array", []):
    print(f"  {row[1]} by {row[2]} ({row[3]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register as Unity Catalog Tool (Preview)
# MAGIC
# MAGIC The vector search index can be registered as a tool in Unity Catalog for use by AI agents.

# COMMAND ----------

# Create the UC function using Python to leverage the variables
search_function_sql = f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.search_song_catalog(
  query STRING COMMENT 'Natural language description of songs to find'
)
RETURNS TABLE (
  song_id STRING,
  title STRING,
  artist_name STRING,
  genre STRING,
  songwriter_names STRING,
  similarity_score DOUBLE
)
COMMENT 'Search the GMR song catalog using semantic similarity. Use this to find songs matching descriptions like "upbeat pop songs" or "acoustic ballads about love".'
RETURN
  SELECT
    song_id,
    title,
    artist_name,
    genre,
    songwriter_names,
    search_score AS similarity_score
  FROM VECTOR_SEARCH(
    index => '{VS_INDEX_NAME}',
    query => query,
    num_results => 10
  )
"""

spark.sql(search_function_sql)
print(f"Created search function: {CATALOG}.{SCHEMA}.search_song_catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook created:
# MAGIC 1. **`song_metadata_embeddings`** - Source table with rich text descriptions and embeddings
# MAGIC 2. **`gmr-vector-search-endpoint`** - Managed endpoint for vector search queries
# MAGIC 3. **`song_metadata_index`** - Delta Sync vector search index
# MAGIC 4. **`search_song_catalog`** - UC function wrapper for agent tool use
# MAGIC
# MAGIC The vector search index enables:
# MAGIC - Semantic song discovery using natural language
# MAGIC - Integration with Mosaic AI agents via UC tools
# MAGIC - Real-time search with automatic Delta sync
# MAGIC
# MAGIC ### Next Steps
# MAGIC Proceed to **05_ai_functions.sql** to explore AI Functions for classification, extraction, and generation.

# COMMAND ----------

songs_count = spark.table(SOURCE_TABLE).count()
print(f"""
Vector Search Setup Complete!
=============================
Endpoint: {VS_ENDPOINT_NAME}
Index: {VS_INDEX_NAME}
Source Table: {SOURCE_TABLE}
Embedding Model: {EMBEDDING_MODEL}

Songs Indexed: {songs_count}

Test the search in the Unity Catalog:
  SELECT * FROM search_song_catalog('upbeat pop songs')
""")
