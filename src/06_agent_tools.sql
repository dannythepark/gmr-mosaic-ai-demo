-- Databricks notebook source
-- MAGIC %md
-- MAGIC # GMR Mosaic AI Demo - Agent Tools (UC Functions)
-- MAGIC
-- MAGIC **Business Context:** This notebook creates the Unity Catalog functions that will serve as
-- MAGIC tools for our Mosaic AI agent. Each function is designed to answer a specific type of
-- MAGIC question that GMR royalty analysts frequently ask.
-- MAGIC
-- MAGIC ## Tools Created
-- MAGIC | Tool | Purpose |
-- MAGIC |------|---------|
-- MAGIC | `lookup_song_royalties` | Get royalty payment history for a song |
-- MAGIC | `search_song_catalog` | Semantic search for songs (uses Vector Search) |
-- MAGIC | `calculate_royalty_split` | Calculate per-songwriter payment breakdown |
-- MAGIC | `get_licensing_summary` | Get licensing stats for a song or artist |
-- MAGIC | `flag_payment_anomaly` | Detect unusual payment patterns |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Configuration

-- COMMAND ----------

USE CATALOG gmr_demo;
USE SCHEMA royalties;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 1: Lookup Song Royalties
-- MAGIC
-- MAGIC This tool retrieves the complete royalty payment history for a song.
-- MAGIC Analysts can search by song title or ISRC code.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gmr_demo.royalties.lookup_song_royalties(
  search_term STRING COMMENT 'Song title or ISRC code to search for'
)
RETURNS TABLE (
  song_id STRING,
  title STRING,
  artist_name STRING,
  isrc_code STRING,
  songwriter_id STRING,
  songwriter_name STRING,
  payment_period STRING,
  gross_amount DOUBLE,
  deductions DOUBLE,
  net_amount DOUBLE,
  payment_status STRING,
  payment_date DATE
)
COMMENT 'Look up royalty payment history for a song by title or ISRC code. Returns all payments associated with the song including songwriter breakdown.'
RETURN
  SELECT
    s.song_id,
    s.title,
    s.artist_name,
    s.isrc_code,
    rp.songwriter_id,
    sw.name AS songwriter_name,
    rp.payment_period,
    rp.gross_amount,
    rp.deductions,
    rp.net_amount,
    rp.payment_status,
    rp.payment_date
  FROM gmr_demo.royalties.songs s
  JOIN gmr_demo.royalties.royalty_payments rp ON s.song_id = rp.song_id
  LEFT JOIN gmr_demo.royalties.songwriters sw ON rp.songwriter_id = sw.songwriter_id
  WHERE LOWER(s.title) LIKE CONCAT('%', LOWER(search_term), '%')
     OR UPPER(s.isrc_code) = UPPER(search_term)
  ORDER BY rp.payment_period DESC, rp.net_amount DESC;

-- COMMAND ----------

-- Test the function
SELECT * FROM lookup_song_royalties('Midnight');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 2: Search Song Catalog (Vector Search)
-- MAGIC
-- MAGIC This tool was created in 04_vector_index.py. It enables semantic search
-- MAGIC of the song catalog using natural language descriptions.

-- COMMAND ----------

-- Verify the search function exists
DESCRIBE FUNCTION gmr_demo.royalties.search_song_catalog;

-- COMMAND ----------

-- Test semantic search
SELECT * FROM search_song_catalog('upbeat pop songs about summer');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 3: Calculate Royalty Split
-- MAGIC
-- MAGIC Given a song and gross royalty amount, calculate how the payment should be
-- MAGIC distributed among the songwriters based on their split percentages.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gmr_demo.royalties.calculate_royalty_split(
  input_song_id STRING COMMENT 'The song_id to calculate splits for',
  gross_amount DOUBLE COMMENT 'The gross royalty amount to distribute'
)
RETURNS TABLE (
  song_id STRING,
  title STRING,
  songwriter_id STRING,
  songwriter_name STRING,
  pro_affiliation STRING,
  split_percentage DOUBLE,
  gross_share DOUBLE,
  estimated_deductions DOUBLE,
  net_share DOUBLE
)
COMMENT 'Calculate the royalty split for a song among its songwriters. Takes a song_id and gross amount, returns per-songwriter breakdown with estimated net payments.'
RETURN
  WITH song_writers AS (
    SELECT
      s.song_id,
      s.title,
      TRIM(writer_id) AS songwriter_id
    FROM gmr_demo.royalties.songs s
    LATERAL VIEW EXPLODE(SPLIT(s.songwriters, ',')) AS writer_id
    WHERE s.song_id = input_song_id
  ),
  writer_details AS (
    SELECT
      sw.song_id,
      sw.title,
      sw.songwriter_id,
      wr.name AS songwriter_name,
      wr.pro_affiliation,
      wr.split_percentage
    FROM song_writers sw
    LEFT JOIN gmr_demo.royalties.songwriters wr ON sw.songwriter_id = wr.songwriter_id
  ),
  normalized_splits AS (
    SELECT
      *,
      split_percentage / SUM(split_percentage) OVER () AS normalized_split
    FROM writer_details
  )
  SELECT
    song_id,
    title,
    songwriter_id,
    songwriter_name,
    pro_affiliation,
    ROUND(normalized_split * 100, 2) AS split_percentage,
    ROUND(gross_amount * normalized_split, 2) AS gross_share,
    ROUND(gross_amount * normalized_split * 0.15, 2) AS estimated_deductions,  -- Assume 15% admin fee
    ROUND(gross_amount * normalized_split * 0.85, 2) AS net_share
  FROM normalized_splits;

-- COMMAND ----------

-- Test royalty split calculation
SELECT * FROM calculate_royalty_split('SONG000001', 10000.00);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 4: Get Licensing Summary
-- MAGIC
-- MAGIC Retrieve licensing statistics for a song or artist, including active license counts,
-- MAGIC revenue by license type, and territory breakdown.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gmr_demo.royalties.get_licensing_summary(
  search_term STRING COMMENT 'Song title, song_id, or artist name to search for'
)
RETURNS TABLE (
  search_match STRING,
  match_type STRING,
  total_active_licenses INT,
  total_license_revenue DOUBLE,
  license_type STRING,
  license_type_count INT,
  license_type_revenue DOUBLE,
  top_territories STRING
)
COMMENT 'Get licensing summary for a song or artist. Returns active license counts, revenue by license type, and territory distribution.'
RETURN
  WITH matched_songs AS (
    SELECT song_id, title, artist_name,
           CASE
             WHEN song_id = search_term THEN 'song_id'
             WHEN LOWER(title) LIKE CONCAT('%', LOWER(search_term), '%') THEN 'title'
             WHEN LOWER(artist_name) LIKE CONCAT('%', LOWER(search_term), '%') THEN 'artist'
           END AS match_type
    FROM gmr_demo.royalties.songs
    WHERE song_id = search_term
       OR LOWER(title) LIKE CONCAT('%', LOWER(search_term), '%')
       OR LOWER(artist_name) LIKE CONCAT('%', LOWER(search_term), '%')
  ),
  license_stats AS (
    SELECT
      COALESCE(ms.title, ms.artist_name, ms.song_id) AS search_match,
      ms.match_type,
      l.license_type,
      l.territory,
      l.fee_amount,
      CASE WHEN l.end_date >= CURRENT_DATE() THEN 1 ELSE 0 END AS is_active
    FROM matched_songs ms
    JOIN gmr_demo.royalties.licenses l ON ms.song_id = l.song_id
  ),
  type_summary AS (
    SELECT
      search_match,
      match_type,
      license_type,
      COUNT(*) AS license_type_count,
      SUM(fee_amount) AS license_type_revenue,
      SUM(is_active) AS active_count
    FROM license_stats
    GROUP BY search_match, match_type, license_type
  ),
  territory_summary AS (
    SELECT
      search_match,
      CONCAT_WS(', ', COLLECT_LIST(territory)) AS top_territories
    FROM (
      SELECT search_match, territory, COUNT(*) AS cnt
      FROM license_stats
      GROUP BY search_match, territory
      ORDER BY cnt DESC
      LIMIT 5
    )
    GROUP BY search_match
  ),
  totals AS (
    SELECT
      search_match,
      SUM(active_count) AS total_active_licenses,
      SUM(license_type_revenue) AS total_license_revenue
    FROM type_summary
    GROUP BY search_match
  )
  SELECT
    ts.search_match,
    ts.match_type,
    t.total_active_licenses,
    t.total_license_revenue,
    ts.license_type,
    ts.license_type_count,
    ts.license_type_revenue,
    ter.top_territories
  FROM type_summary ts
  JOIN totals t ON ts.search_match = t.search_match
  LEFT JOIN territory_summary ter ON ts.search_match = ter.search_match
  ORDER BY ts.license_type_revenue DESC;

-- COMMAND ----------

-- Test licensing summary
SELECT * FROM get_licensing_summary('SONG000001');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 5: Flag Payment Anomaly
-- MAGIC
-- MAGIC Detect unusual payments that deviate significantly from historical patterns.
-- MAGIC A payment is flagged if it's more than 2 standard deviations from the mean.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gmr_demo.royalties.flag_payment_anomaly(
  input_song_id STRING COMMENT 'The song_id to check for anomalies',
  input_licensee STRING COMMENT 'The licensee name to check (optional, use NULL for all)'
)
RETURNS TABLE (
  song_id STRING,
  title STRING,
  licensee_name STRING,
  payment_id STRING,
  payment_period STRING,
  net_amount DOUBLE,
  historical_mean DOUBLE,
  historical_stddev DOUBLE,
  z_score DOUBLE,
  is_anomaly BOOLEAN,
  anomaly_type STRING
)
COMMENT 'Check if payments for a song/licensee combination show anomalies. Flags payments that deviate more than 2 standard deviations from historical average.'
RETURN
  WITH song_info AS (
    SELECT song_id, title FROM gmr_demo.royalties.songs WHERE song_id = input_song_id
  ),
  payment_history AS (
    SELECT
      rp.song_id,
      l.licensee_name,
      rp.payment_id,
      rp.payment_period,
      rp.net_amount
    FROM gmr_demo.royalties.royalty_payments rp
    JOIN gmr_demo.royalties.licenses l ON rp.song_id = l.song_id
    WHERE rp.song_id = input_song_id
      AND (input_licensee IS NULL OR l.licensee_name = input_licensee)
  ),
  stats AS (
    SELECT
      song_id,
      licensee_name,
      AVG(net_amount) AS historical_mean,
      STDDEV(net_amount) AS historical_stddev
    FROM payment_history
    GROUP BY song_id, licensee_name
  ),
  anomaly_check AS (
    SELECT
      ph.song_id,
      si.title,
      ph.licensee_name,
      ph.payment_id,
      ph.payment_period,
      ph.net_amount,
      s.historical_mean,
      s.historical_stddev,
      CASE
        WHEN s.historical_stddev > 0
        THEN (ph.net_amount - s.historical_mean) / s.historical_stddev
        ELSE 0
      END AS z_score
    FROM payment_history ph
    JOIN song_info si ON ph.song_id = si.song_id
    JOIN stats s ON ph.song_id = s.song_id AND ph.licensee_name = s.licensee_name
  )
  SELECT
    song_id,
    title,
    licensee_name,
    payment_id,
    payment_period,
    ROUND(net_amount, 2) AS net_amount,
    ROUND(historical_mean, 2) AS historical_mean,
    ROUND(historical_stddev, 2) AS historical_stddev,
    ROUND(z_score, 2) AS z_score,
    ABS(z_score) > 2 AS is_anomaly,
    CASE
      WHEN z_score > 2 THEN 'UNUSUALLY_HIGH'
      WHEN z_score < -2 THEN 'UNUSUALLY_LOW'
      ELSE 'NORMAL'
    END AS anomaly_type
  FROM anomaly_check
  ORDER BY ABS(z_score) DESC;

-- COMMAND ----------

-- Test anomaly detection
SELECT * FROM flag_payment_anomaly('SONG000001', NULL)
WHERE is_anomaly = TRUE;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Verify All Tools Are Registered

-- COMMAND ----------

-- List all functions in the schema
SHOW FUNCTIONS IN gmr_demo.royalties LIKE '*';

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool Descriptions for Agent
-- MAGIC
-- MAGIC These descriptions will be used by the Mosaic AI agent to understand when to call each tool:
-- MAGIC
-- MAGIC | Tool | When to Use |
-- MAGIC |------|-------------|
-- MAGIC | `lookup_song_royalties` | When user asks about royalty payments, payment history, or earnings for a specific song |
-- MAGIC | `search_song_catalog` | When user wants to find songs by description, genre, mood, or semantic similarity |
-- MAGIC | `calculate_royalty_split` | When user wants to know how a payment should be divided among songwriters |
-- MAGIC | `get_licensing_summary` | When user asks about licensing deals, territories, or license revenue |
-- MAGIC | `flag_payment_anomaly` | When user wants to check for unusual payments or potential issues with a licensee |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC This notebook created 5 Unity Catalog functions that will serve as tools for our Mosaic AI agent:
-- MAGIC
-- MAGIC 1. **`lookup_song_royalties`** - Query royalty payment history by song title or ISRC
-- MAGIC 2. **`search_song_catalog`** - Semantic search using Vector Search index
-- MAGIC 3. **`calculate_royalty_split`** - Compute songwriter payment breakdown
-- MAGIC 4. **`get_licensing_summary`** - Aggregate licensing statistics
-- MAGIC 5. **`flag_payment_anomaly`** - Statistical anomaly detection for payments
-- MAGIC
-- MAGIC All functions are registered in Unity Catalog and can be called directly in SQL or
-- MAGIC programmatically via the Unity Catalog Functions API.
-- MAGIC
-- MAGIC ### Next Steps
-- MAGIC Proceed to **07_build_agent.py** to construct the Mosaic AI agent using these tools.
