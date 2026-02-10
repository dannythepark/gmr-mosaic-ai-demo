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

dbutils.widgets.text("catalog", "gmr_demo", "Catalog Name")
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
# MAGIC ## Generate Embeddings
# MAGIC
# MAGIC Use the Foundation Model endpoint `databricks-gte-large-en` to generate
# MAGIC 1024-dimensional embeddings for each song's metadata text.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import mlflow.deployments

# Initialize the deployments client for Foundation Model APIs
client = mlflow.deployments.get_deploy_client("databricks")

def get_embeddings(texts):
    """Generate embeddings using the Foundation Model endpoint."""
    response = client.predict(
        endpoint=EMBEDDING_MODEL,
        inputs={"input": texts}
    )
    return [item["embedding"] for item in response["data"]]

# COMMAND ----------

# Convert to Pandas for embedding generation (for smaller datasets)
# For large datasets, use Spark UDFs or batch processing
songs_pd = song_metadata_df.toPandas()

print(f"Generating embeddings for {len(songs_pd)} songs...")

# COMMAND ----------

# Generate embeddings in batches
BATCH_SIZE = 50
all_embeddings = []

for i in range(0, len(songs_pd), BATCH_SIZE):
    batch_texts = songs_pd["metadata_text"].iloc[i:i+BATCH_SIZE].tolist()
    batch_embeddings = get_embeddings(batch_texts)
    all_embeddings.extend(batch_embeddings)
    print(f"Processed {min(i+BATCH_SIZE, len(songs_pd))}/{len(songs_pd)} songs")

songs_pd["embedding"] = all_embeddings

# COMMAND ----------

# Convert back to Spark DataFrame
from pyspark.sql.types import ArrayType, FloatType

# Create schema for the embedding column
embedding_schema = ArrayType(FloatType())

# Convert to Spark DataFrame
songs_with_embeddings = spark.createDataFrame(songs_pd)

# Write to Delta table
songs_with_embeddings.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SOURCE_TABLE)

print(f"Embedding source table created: {SOURCE_TABLE}")

# COMMAND ----------

# Verify the embeddings
display(
    spark.table(SOURCE_TABLE)
    .select("song_id", "title", "genre", F.size("embedding").alias("embedding_dim"))
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
        endpoint = vsc.get_endpoint(endpoint_name)
        status = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")
        if status == "ONLINE":
            print(f"Endpoint '{endpoint_name}' is ONLINE")
            return True
        print(f"Endpoint status: {status}. Waiting...")
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

# Create the Delta Sync index
index = vsc.create_delta_sync_index(
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME,
    source_table_name=SOURCE_TABLE,
    pipeline_type="TRIGGERED",  # Manual sync trigger
    primary_key="song_id",
    embedding_dimension=EMBEDDING_DIM,
    embedding_vector_column="embedding",
    # Columns to include in search results
    columns_to_sync=["song_id", "title", "artist_name", "genre", "songwriter_names", "publisher", "metadata_text"]
)

print(f"Vector Search index created: {VS_INDEX_NAME}")

# COMMAND ----------

# Wait for index to be ready
def wait_for_index(endpoint_name, index_name, timeout=1200):
    """Wait for index to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            index = vsc.get_index(endpoint_name, index_name)
            status = index.get("status", {}).get("detailed_state", "UNKNOWN")
            if status == "ONLINE":
                print(f"Index '{index_name}' is ONLINE")
                return True
            print(f"Index status: {status}. Waiting...")
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
    # Get embedding for the query
    query_embedding = get_embeddings([query_text])[0]

    # Search the index
    index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    results = index.similarity_search(
        query_vector=query_embedding,
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

# MAGIC %sql
# MAGIC -- Create a function that wraps the vector search for agent tool use
# MAGIC CREATE OR REPLACE FUNCTION gmr_demo.royalties.search_song_catalog(
# MAGIC   query STRING COMMENT 'Natural language description of songs to find'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC   song_id STRING,
# MAGIC   title STRING,
# MAGIC   artist_name STRING,
# MAGIC   genre STRING,
# MAGIC   songwriter_names STRING,
# MAGIC   similarity_score DOUBLE
# MAGIC )
# MAGIC COMMENT 'Search the GMR song catalog using semantic similarity. Use this to find songs matching descriptions like "upbeat pop songs" or "acoustic ballads about love".'
# MAGIC RETURN
# MAGIC   SELECT
# MAGIC     song_id,
# MAGIC     title,
# MAGIC     artist_name,
# MAGIC     genre,
# MAGIC     songwriter_names,
# MAGIC     score AS similarity_score
# MAGIC   FROM VECTOR_SEARCH(
# MAGIC     index => 'gmr_demo.royalties.song_metadata_index',
# MAGIC     query => query,
# MAGIC     num_results => 10
# MAGIC   );

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

print(f"""
Vector Search Setup Complete!
=============================
Endpoint: {VS_ENDPOINT_NAME}
Index: {VS_INDEX_NAME}
Source Table: {SOURCE_TABLE}
Embedding Model: {EMBEDDING_MODEL}
Embedding Dimensions: {EMBEDDING_DIM}

Songs Indexed: {songs_pd.shape[0]}

Test the search in the Unity Catalog:
  SELECT * FROM search_song_catalog('upbeat pop songs')
""")
