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
-- MAGIC | `lookup_songwriter_earnings` | Get earnings summary for a songwriter |
-- MAGIC | `get_top_royalty_songs` | Ranked list of highest-earning songs |
-- MAGIC | `get_catalog_overview` | High-level portfolio statistics |
-- MAGIC | `get_revenue_by_territory` | Geographic revenue breakdown by territory |
-- MAGIC | `get_platform_performance` | Streaming/radio platform analytics |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Configuration

-- COMMAND ----------

USE CATALOG gmr_demo_catalog;
USE SCHEMA royalties;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 1: Lookup Song Royalties
-- MAGIC
-- MAGIC This tool retrieves the complete royalty payment history for a song.
-- MAGIC Analysts can search by song title or ISRC code.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gmr_demo_catalog.royalties.lookup_song_royalties(
  search_term STRING COMMENT 'Song ID (e.g. SONG000001), song title, or ISRC code to search for'
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
COMMENT 'Look up royalty payment history for a song by song_id, title, or ISRC code. Returns all payments associated with the song including songwriter breakdown.'
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
  FROM gmr_demo_catalog.royalties.songs s
  JOIN gmr_demo_catalog.royalties.royalty_payments rp ON s.song_id = rp.song_id
  LEFT JOIN gmr_demo_catalog.royalties.songwriters sw ON rp.songwriter_id = sw.songwriter_id
  WHERE s.song_id = search_term
     OR LOWER(s.title) LIKE CONCAT('%', LOWER(search_term), '%')
     OR UPPER(s.isrc_code) = UPPER(search_term)
  ORDER BY rp.payment_period DESC, rp.net_amount DESC;

-- COMMAND ----------

-- Test the function
SELECT * FROM lookup_song_royalties('Bohemian Rhapsody');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 2: Search Song Catalog (Vector Search)
-- MAGIC
-- MAGIC This tool was created in 04_vector_index.py. It enables semantic search
-- MAGIC of the song catalog using natural language descriptions.

-- COMMAND ----------

-- Verify the search function exists
DESCRIBE FUNCTION gmr_demo_catalog.royalties.search_song_catalog;

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

CREATE OR REPLACE FUNCTION gmr_demo_catalog.royalties.calculate_royalty_split(
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
    FROM gmr_demo_catalog.royalties.songs s
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
    LEFT JOIN gmr_demo_catalog.royalties.songwriters wr ON sw.songwriter_id = wr.songwriter_id
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

CREATE OR REPLACE FUNCTION gmr_demo_catalog.royalties.get_licensing_summary(
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
    FROM gmr_demo_catalog.royalties.songs
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
    JOIN gmr_demo_catalog.royalties.licenses l ON ms.song_id = l.song_id
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

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 6: Lookup Songwriter Earnings
-- MAGIC
-- MAGIC Get earnings summary for a specific songwriter by ID or name.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gmr_demo_catalog.royalties.lookup_songwriter_earnings(
  search_term STRING COMMENT 'Songwriter ID (e.g. SW00001) or songwriter name to search for'
)
RETURNS TABLE (
  songwriter_id STRING,
  songwriter_name STRING,
  pro_affiliation STRING,
  payment_period STRING,
  total_gross DOUBLE,
  total_deductions DOUBLE,
  total_net DOUBLE,
  num_payments INT,
  num_songs INT
)
COMMENT 'Look up earnings for a songwriter by ID or name. Returns payment totals grouped by period.'
RETURN
  SELECT
    sw.songwriter_id,
    sw.name AS songwriter_name,
    sw.pro_affiliation,
    rp.payment_period,
    ROUND(SUM(rp.gross_amount), 2) AS total_gross,
    ROUND(SUM(rp.deductions), 2) AS total_deductions,
    ROUND(SUM(rp.net_amount), 2) AS total_net,
    COUNT(*) AS num_payments,
    COUNT(DISTINCT rp.song_id) AS num_songs
  FROM gmr_demo_catalog.royalties.songwriters sw
  JOIN gmr_demo_catalog.royalties.royalty_payments rp ON sw.songwriter_id = rp.songwriter_id
  WHERE sw.songwriter_id = search_term
     OR LOWER(sw.name) LIKE CONCAT('%', LOWER(search_term), '%')
  GROUP BY sw.songwriter_id, sw.name, sw.pro_affiliation, rp.payment_period
  ORDER BY rp.payment_period DESC, total_net DESC;

-- COMMAND ----------

-- Test songwriter earnings lookup
SELECT * FROM lookup_songwriter_earnings('SW00001');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 7: Get Top Royalty Songs
-- MAGIC
-- MAGIC Retrieve the highest-earning songs by total net royalty payments.
-- MAGIC Supports filtering by genre, artist, or time period.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gmr_demo_catalog.royalties.get_top_royalty_songs(
  num_results INT COMMENT 'Number of top songs to return (default 10, max 50)',
  filter_genre STRING COMMENT 'Optional genre filter (e.g. Pop, Rock, Hip-Hop). Pass NULL or omit for all genres.',
  filter_period STRING COMMENT 'Optional payment period filter (e.g. 2025-Q1). Pass NULL or omit for all periods.'
)
RETURNS TABLE (
  rank INT,
  song_id STRING,
  title STRING,
  artist_name STRING,
  genre STRING,
  total_gross DOUBLE,
  total_deductions DOUBLE,
  total_net DOUBLE,
  num_payments INT,
  num_songwriters INT
)
COMMENT 'Get the top royalty-earning songs ranked by total net payments. Use this when the user asks about top songs, highest-earning songs, best-performing songs, or royalty rankings. Optionally filter by genre or payment period.'
RETURN
  WITH ranked AS (
    SELECT
      ROW_NUMBER() OVER (ORDER BY SUM(rp.net_amount) DESC) AS rank,
      s.song_id,
      s.title,
      s.artist_name,
      s.genre,
      ROUND(SUM(rp.gross_amount), 2) AS total_gross,
      ROUND(SUM(rp.deductions), 2) AS total_deductions,
      ROUND(SUM(rp.net_amount), 2) AS total_net,
      COUNT(*) AS num_payments,
      COUNT(DISTINCT rp.songwriter_id) AS num_songwriters
    FROM gmr_demo_catalog.royalties.songs s
    JOIN gmr_demo_catalog.royalties.royalty_payments rp ON s.song_id = rp.song_id
    WHERE (filter_genre IS NULL OR UPPER(filter_genre) = 'NULL' OR LOWER(s.genre) = LOWER(filter_genre))
      AND (filter_period IS NULL OR UPPER(filter_period) = 'NULL' OR rp.payment_period = filter_period)
    GROUP BY s.song_id, s.title, s.artist_name, s.genre
  )
  SELECT * FROM ranked
  WHERE rank <= COALESCE(num_results, 10);

-- COMMAND ----------

-- Test top royalty songs
SELECT * FROM get_top_royalty_songs(5, NULL, NULL);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 8: Get Catalog Overview
-- MAGIC
-- MAGIC Provide high-level portfolio statistics for the entire catalog or filtered by genre/artist.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gmr_demo_catalog.royalties.get_catalog_overview(
  filter_genre STRING COMMENT 'Optional genre filter (e.g. Pop, Rock). Pass NULL or omit for entire catalog.'
)
RETURNS TABLE (
  total_songs INT,
  total_artists INT,
  total_songwriters INT,
  total_genres INT,
  total_gross_revenue DOUBLE,
  total_net_revenue DOUBLE,
  total_licenses INT,
  active_licenses INT,
  avg_royalty_per_song DOUBLE,
  top_genre STRING,
  top_artist STRING
)
COMMENT 'Get high-level overview of the music catalog and royalty portfolio. Use this when the user asks about overall catalog stats, portfolio summary, how many songs/artists, total revenue, or general overview questions. Optionally filter by genre.'
RETURN
  WITH song_base AS (
    SELECT * FROM gmr_demo_catalog.royalties.songs
    WHERE (filter_genre IS NULL OR UPPER(filter_genre) = 'NULL' OR LOWER(genre) = LOWER(filter_genre))
  ),
  revenue AS (
    SELECT
      ROUND(SUM(rp.gross_amount), 2) AS total_gross_revenue,
      ROUND(SUM(rp.net_amount), 2) AS total_net_revenue
    FROM gmr_demo_catalog.royalties.royalty_payments rp
    WHERE rp.song_id IN (SELECT song_id FROM song_base)
  ),
  license_stats AS (
    SELECT
      COUNT(*) AS total_licenses,
      SUM(CASE WHEN end_date >= CURRENT_DATE() THEN 1 ELSE 0 END) AS active_licenses
    FROM gmr_demo_catalog.royalties.licenses l
    WHERE l.song_id IN (SELECT song_id FROM song_base)
  ),
  top_genre AS (
    SELECT genre AS top_genre
    FROM (
      SELECT s.genre, SUM(rp.net_amount) AS rev
      FROM song_base s
      JOIN gmr_demo_catalog.royalties.royalty_payments rp ON s.song_id = rp.song_id
      GROUP BY s.genre ORDER BY rev DESC LIMIT 1
    )
  ),
  top_artist AS (
    SELECT artist_name AS top_artist
    FROM (
      SELECT s.artist_name, SUM(rp.net_amount) AS rev
      FROM song_base s
      JOIN gmr_demo_catalog.royalties.royalty_payments rp ON s.song_id = rp.song_id
      GROUP BY s.artist_name ORDER BY rev DESC LIMIT 1
    )
  )
  SELECT
    (SELECT COUNT(*) FROM song_base) AS total_songs,
    (SELECT COUNT(DISTINCT artist_name) FROM song_base) AS total_artists,
    (SELECT COUNT(DISTINCT songwriter_id) FROM gmr_demo_catalog.royalties.songwriters) AS total_songwriters,
    (SELECT COUNT(DISTINCT genre) FROM song_base) AS total_genres,
    r.total_gross_revenue,
    r.total_net_revenue,
    ls.total_licenses,
    ls.active_licenses,
    ROUND(r.total_net_revenue / NULLIF((SELECT COUNT(*) FROM song_base), 0), 2) AS avg_royalty_per_song,
    tg.top_genre,
    ta.top_artist
  FROM revenue r
  CROSS JOIN license_stats ls
  CROSS JOIN top_genre tg
  CROSS JOIN top_artist ta;

-- COMMAND ----------

-- Test catalog overview
SELECT * FROM get_catalog_overview(NULL);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 9: Get Revenue by Territory
-- MAGIC
-- MAGIC Analyze licensing revenue distribution across geographic territories.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gmr_demo_catalog.royalties.get_revenue_by_territory(
  filter_territory STRING COMMENT 'Optional territory filter (e.g. US, UK, DE). Pass NULL or omit for all territories.'
)
RETURNS TABLE (
  territory STRING,
  total_licenses INT,
  active_licenses INT,
  total_revenue DOUBLE,
  avg_fee_per_license DOUBLE,
  top_license_type STRING,
  top_licensee STRING,
  num_songs INT
)
COMMENT 'Get licensing revenue breakdown by geographic territory. Use this when the user asks about revenue by country, territory performance, geographic distribution, international licensing, or market analysis. Optionally filter to a specific territory code (US, UK, DE, JP, etc.).'
RETURN
  WITH territory_base AS (
    SELECT *
    FROM gmr_demo_catalog.royalties.licenses
    WHERE (filter_territory IS NULL OR UPPER(filter_territory) = 'NULL' OR UPPER(territory) = UPPER(filter_territory))
  ),
  territory_stats AS (
    SELECT
      territory,
      COUNT(*) AS total_licenses,
      SUM(CASE WHEN end_date >= CURRENT_DATE() THEN 1 ELSE 0 END) AS active_licenses,
      ROUND(SUM(fee_amount), 2) AS total_revenue,
      ROUND(AVG(fee_amount), 2) AS avg_fee_per_license,
      COUNT(DISTINCT song_id) AS num_songs
    FROM territory_base
    GROUP BY territory
  ),
  top_types AS (
    SELECT territory, license_type AS top_license_type
    FROM (
      SELECT territory, license_type, SUM(fee_amount) AS rev,
             ROW_NUMBER() OVER (PARTITION BY territory ORDER BY SUM(fee_amount) DESC) AS rn
      FROM territory_base
      GROUP BY territory, license_type
    ) WHERE rn = 1
  ),
  top_licensees AS (
    SELECT territory, licensee_name AS top_licensee
    FROM (
      SELECT territory, licensee_name, SUM(fee_amount) AS rev,
             ROW_NUMBER() OVER (PARTITION BY territory ORDER BY SUM(fee_amount) DESC) AS rn
      FROM territory_base
      GROUP BY territory, licensee_name
    ) WHERE rn = 1
  )
  SELECT
    ts.territory,
    ts.total_licenses,
    ts.active_licenses,
    ts.total_revenue,
    ts.avg_fee_per_license,
    tt.top_license_type,
    tl.top_licensee,
    ts.num_songs
  FROM territory_stats ts
  LEFT JOIN top_types tt ON ts.territory = tt.territory
  LEFT JOIN top_licensees tl ON ts.territory = tl.territory
  ORDER BY ts.total_revenue DESC;

-- COMMAND ----------

-- Test territory revenue
SELECT * FROM get_revenue_by_territory(NULL);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Tool 10: Get Platform Performance
-- MAGIC
-- MAGIC Analyze song performance across streaming platforms and radio stations.

-- COMMAND ----------

CREATE OR REPLACE FUNCTION gmr_demo_catalog.royalties.get_platform_performance(
  search_term STRING COMMENT 'Song title, song_id, or artist name. Use NULL for all songs.',
  filter_platform STRING COMMENT 'Optional platform filter (e.g. Spotify, Apple Music). Pass NULL or omit for all platforms.'
)
RETURNS TABLE (
  platform STRING,
  total_plays BIGINT,
  total_play_hours DOUBLE,
  num_songs INT,
  num_territories INT,
  avg_plays_per_song DOUBLE,
  top_song_title STRING,
  top_song_plays BIGINT
)
COMMENT 'Get performance analytics across streaming platforms and radio stations. Use this when the user asks about streaming numbers, play counts, platform performance, which platform has the most plays, Spotify vs Apple Music, or radio airplay. Optionally filter by song/artist and platform.'
RETURN
  WITH filtered_songs AS (
    SELECT song_id, title, artist_name
    FROM gmr_demo_catalog.royalties.songs
    WHERE search_term IS NULL OR UPPER(search_term) = 'NULL'
       OR song_id = search_term
       OR LOWER(title) LIKE CONCAT('%', LOWER(search_term), '%')
       OR LOWER(artist_name) LIKE CONCAT('%', LOWER(search_term), '%')
  ),
  play_data AS (
    SELECT
      pl.platform,
      pl.song_id,
      fs.title,
      pl.territory,
      pl.duration_played
    FROM gmr_demo_catalog.royalties.performance_logs pl
    JOIN filtered_songs fs ON pl.song_id = fs.song_id
    WHERE (filter_platform IS NULL OR UPPER(filter_platform) = 'NULL' OR LOWER(pl.platform) = LOWER(filter_platform))
  ),
  platform_stats AS (
    SELECT
      platform,
      COUNT(*) AS total_plays,
      ROUND(SUM(duration_played) / 3600.0, 1) AS total_play_hours,
      COUNT(DISTINCT song_id) AS num_songs,
      COUNT(DISTINCT territory) AS num_territories,
      ROUND(COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT song_id), 0), 1) AS avg_plays_per_song
    FROM play_data
    GROUP BY platform
  ),
  top_songs AS (
    SELECT platform, title AS top_song_title, cnt AS top_song_plays
    FROM (
      SELECT platform, title, COUNT(*) AS cnt,
             ROW_NUMBER() OVER (PARTITION BY platform ORDER BY COUNT(*) DESC) AS rn
      FROM play_data
      GROUP BY platform, title
    ) WHERE rn = 1
  )
  SELECT
    ps.platform,
    ps.total_plays,
    ps.total_play_hours,
    ps.num_songs,
    ps.num_territories,
    ps.avg_plays_per_song,
    ts.top_song_title,
    ts.top_song_plays
  FROM platform_stats ps
  LEFT JOIN top_songs ts ON ps.platform = ts.platform
  ORDER BY ps.total_plays DESC;

-- COMMAND ----------

-- Test platform performance
SELECT * FROM get_platform_performance(NULL, NULL);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Verify All Tools Are Registered

-- COMMAND ----------

-- List all functions in the schema
SHOW FUNCTIONS IN gmr_demo_catalog.royalties LIKE '*';

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
-- MAGIC | `lookup_songwriter_earnings` | When user asks about a songwriter's earnings, payments, or income by songwriter ID or name |
-- MAGIC | `get_top_royalty_songs` | When user asks about top songs, highest earners, best performers, or royalty rankings |
-- MAGIC | `get_catalog_overview` | When user asks about overall portfolio stats, total revenue, how many songs/artists |
-- MAGIC | `get_revenue_by_territory` | When user asks about revenue by country, geographic distribution, territory performance |
-- MAGIC | `get_platform_performance` | When user asks about streaming numbers, play counts, platform comparisons |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC This notebook created 9 Unity Catalog functions that will serve as tools for our Mosaic AI agent:
-- MAGIC
-- MAGIC 1. **`lookup_song_royalties`** - Query royalty payment history by song_id, title, or ISRC
-- MAGIC 2. **`search_song_catalog`** - Semantic search using Vector Search index
-- MAGIC 3. **`calculate_royalty_split`** - Compute songwriter payment breakdown
-- MAGIC 4. **`get_licensing_summary`** - Aggregate licensing statistics
-- MAGIC 5. **`lookup_songwriter_earnings`** - Query songwriter earnings by ID or name
-- MAGIC 6. **`get_top_royalty_songs`** - Ranked list of highest-earning songs
-- MAGIC 7. **`get_catalog_overview`** - High-level portfolio statistics
-- MAGIC 8. **`get_revenue_by_territory`** - Geographic revenue breakdown
-- MAGIC 9. **`get_platform_performance`** - Streaming/radio performance analytics
-- MAGIC
-- MAGIC All functions are registered in Unity Catalog and can be called directly in SQL or
-- MAGIC programmatically via the Unity Catalog Functions API.
-- MAGIC
-- MAGIC ### Next Steps
-- MAGIC Proceed to **07_build_agent.py** to construct the Mosaic AI agent using these tools.
