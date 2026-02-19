-- Databricks notebook source
-- MAGIC %md
-- MAGIC # GMR Mosaic AI Demo - AI Functions in SQL
-- MAGIC
-- MAGIC **Business Context:** As a data analyst at GMR, you can now leverage Large Language Models
-- MAGIC directly in SQL queries using Databricks AI Functions. No Python required! This enables:
-- MAGIC - **Mood classification** for sync licensing opportunities
-- MAGIC - **Entity extraction** from license documents
-- MAGIC - **Automated summarization** of royalty disputes
-- MAGIC - **Sentiment analysis** on partner communications
-- MAGIC
-- MAGIC ## AI Functions Available
-- MAGIC | Function | Purpose |
-- MAGIC |----------|---------|
-- MAGIC | `ai_classify()` | Classify text into predefined categories |
-- MAGIC | `ai_extract()` | Extract structured entities from unstructured text |
-- MAGIC | `ai_query()` | Free-form LLM queries for generation and analysis |
-- MAGIC | `ai_similarity()` | Compute semantic similarity between texts |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Configuration

-- COMMAND ----------

USE CATALOG gmr_demo_catalog;
USE SCHEMA royalties;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. Mood Classification with `ai_classify`
-- MAGIC
-- MAGIC Classify songs by mood for sync licensing opportunities. Music supervisors searching
-- MAGIC for songs for TV, film, or commercials need to find tracks by emotional tone, not just genre.
-- MAGIC This adds a dimension that doesn't exist in the catalog today.

-- COMMAND ----------

-- Classify songs by mood for sync licensing
SELECT
  song_id,
  title,
  artist_name,
  genre,
  ai_classify(
    CONCAT('Song: "', title, '" by ', artist_name, ' (Genre: ', genre, ')'),
    ARRAY('Upbeat', 'Melancholy', 'Energetic', 'Romantic', 'Dark', 'Chill')
  ) AS mood
FROM songs
LIMIT 20;

-- COMMAND ----------

-- Find romantic songs across all genres for a sync licensing request
SELECT
  song_id,
  title,
  artist_name,
  genre,
  ai_classify(
    CONCAT('Song: "', title, '" by ', artist_name, ' (Genre: ', genre, ')'),
    ARRAY('Upbeat', 'Melancholy', 'Energetic', 'Romantic', 'Dark', 'Chill')
  ) AS mood
FROM songs
WHERE ai_classify(
    CONCAT('Song: "', title, '" by ', artist_name, ' (Genre: ', genre, ')'),
    ARRAY('Upbeat', 'Melancholy', 'Energetic', 'Romantic', 'Dark', 'Chill')
  ) = 'Romantic'
LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Entity Extraction with `ai_extract`
-- MAGIC
-- MAGIC Extract structured information from unstructured license documents.
-- MAGIC First, let's create a sample table with license text documents.

-- COMMAND ----------

-- Create a sample table with license document text
CREATE OR REPLACE TEMPORARY VIEW raw_license_docs AS
SELECT
  'DOC001' AS doc_id,
  'This License Agreement ("Agreement") is entered into as of January 15, 2025, by and between Global Music Rights LLC ("Licensor") and Spotify USA Inc. ("Licensee"). The Licensee shall pay a fee of $50,000.00 USD for the territory of United States and Canada. This agreement shall be effective from January 15, 2025 through January 14, 2028.' AS license_text
UNION ALL
SELECT
  'DOC002',
  'Performance License granted to iHeartMedia Inc. for broadcast rights across the European Union territories. License fee: €75,000 EUR annually. Term: March 1, 2025 to February 28, 2027. Includes radio and digital streaming rights.'
UNION ALL
SELECT
  'DOC003',
  'Live Venue License Agreement with Live Nation Entertainment covering Madison Square Garden and affiliated venues. Coverage: North America. Annual fee: $125,000 USD. Effective period: June 1, 2025 - May 31, 2026.';

-- COMMAND ----------

-- Extract entities from license documents
SELECT
  doc_id,
  ai_extract(
    license_text,
    ARRAY('licensee_name', 'territory', 'effective_date', 'fee_amount', 'license_type')
  ) AS extracted_entities,
  license_text
FROM raw_license_docs;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3. Text Generation with `ai_query`
-- MAGIC
-- MAGIC Generate human-readable summaries, explanations, and analysis using LLMs.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Generate Executive Summaries for Royalty Disputes

-- COMMAND ----------

-- Create sample royalty disputes table
CREATE OR REPLACE TEMPORARY VIEW royalty_disputes AS
SELECT
  'DISP001' AS dispute_id,
  'SONG000042' AS song_id,
  'StreamCo Inc.' AS licensee,
  'Payment discrepancy identified. Reported plays: 1,245,000. Calculated royalty at $0.004/play = $4,980. Received payment: $3,200. Difference: $1,780. StreamCo claims technical issue with play counter during Q3 2025. Internal audit shows consistent underpayment pattern over 3 quarters. Previous disputes: Q1 2025 ($890 discrepancy), Q2 2025 ($1,200 discrepancy). Licensee has requested payment plan for backpay.' AS dispute_notes
UNION ALL
SELECT
  'DISP002',
  'SONG000089',
  'RadioMax Broadcasting',
  'Territorial license violation detected. Song played in Germany (DE) but license only covers US and CA territories. Total unauthorized plays: 45,000. Estimated damages: $2,250 based on standard licensing rates. RadioMax acknowledges the violation, citing syndication partner error. Proposed resolution: retroactive license purchase plus 10% penalty.';

-- COMMAND ----------

-- Generate executive summaries for disputes
SELECT
  dispute_id,
  song_id,
  licensee,
  ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    CONCAT(
      'You are a music royalty analyst. Summarize this royalty dispute in 2-3 sentences for a non-technical executive. Focus on the financial impact and recommended action.\n\nDispute Details:\n',
      dispute_notes
    )
  ) AS executive_summary
FROM royalty_disputes;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Generate Songwriter Payment Notifications

-- COMMAND ----------

-- Generate personalized payment notifications
SELECT
  sw.songwriter_id,
  sw.name,
  sw.email,
  SUM(rp.net_amount) AS total_payment,
  ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    CONCAT(
      'Write a brief, professional payment notification email for songwriter ',
      sw.name,
      '. Total payment amount: $',
      CAST(SUM(rp.net_amount) AS STRING),
      '. Payment period: Q4 2025. Keep it under 50 words. Be warm but professional.'
    )
  ) AS notification_email
FROM songwriters sw
JOIN royalty_payments rp ON sw.songwriter_id = rp.songwriter_id
WHERE rp.payment_period = '2025-Q4'
  AND rp.payment_status = 'completed'
GROUP BY sw.songwriter_id, sw.name, sw.email
LIMIT 5;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4. Sentiment Analysis on Partner Communications

-- COMMAND ----------

-- Create sample partner emails
CREATE OR REPLACE TEMPORARY VIEW partner_emails AS
SELECT
  'EMAIL001' AS email_id,
  'Spotify USA Inc.' AS partner,
  'Thank you for the quick resolution on the licensing renewal. We appreciate the flexibility on payment terms and look forward to continuing our partnership. The new dashboard access has been incredibly helpful for our team.' AS email_body
UNION ALL
SELECT
  'EMAIL002',
  'RadioMax Broadcasting',
  'We are extremely disappointed with the recent audit findings. The penalties seem excessive and we request an immediate review. Our legal team will be reaching out to discuss the territorial violation claims. This matter requires urgent attention.'
UNION ALL
SELECT
  'EMAIL003',
  'iHeartMedia Inc.',
  'Following up on our previous conversation about the Q3 royalty statement. We noticed a few minor discrepancies that we would like to clarify. Please advise on the best time for a call this week.'
UNION ALL
SELECT
  'EMAIL004',
  'Live Nation Entertainment',
  'URGENT: The venue reporting system has been down for 48 hours and we cannot submit play counts. This will significantly delay our compliance reporting. We need immediate technical support or risk missing the deadline.';

-- COMMAND ----------

-- Analyze sentiment of partner communications
SELECT
  email_id,
  partner,
  ai_classify(
    email_body,
    ARRAY('positive', 'neutral', 'negative', 'escalation')
  ) AS sentiment,
  email_body
FROM partner_emails;

-- COMMAND ----------

-- Flag emails requiring immediate attention
SELECT
  email_id,
  partner,
  ai_classify(email_body, ARRAY('positive', 'neutral', 'negative', 'escalation')) AS sentiment,
  ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    CONCAT(
      'In one sentence, summarize the main issue and urgency level of this partner communication:\n\n',
      email_body
    )
  ) AS issue_summary
FROM partner_emails
WHERE ai_classify(email_body, ARRAY('positive', 'neutral', 'negative', 'escalation')) IN ('negative', 'escalation');

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5. Similarity Analysis with `ai_similarity`
-- MAGIC
-- MAGIC Find similar songs based on their metadata descriptions.

-- COMMAND ----------

-- Find songs similar to a reference song
WITH reference_song AS (
  SELECT metadata_text
  FROM song_metadata_embeddings
  WHERE title = 'Bohemian Rhapsody'
  LIMIT 1
)
SELECT
  s.song_id,
  s.title,
  s.artist_name,
  s.genre,
  ai_similarity(s.metadata_text, r.metadata_text) AS similarity_score
FROM song_metadata_embeddings s
CROSS JOIN reference_song r
WHERE s.title != 'Bohemian Rhapsody'
ORDER BY similarity_score DESC
LIMIT 10;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 6. Batch Mood Tagging for Sync Licensing

-- COMMAND ----------

-- Mood distribution across the catalog - what's our sync licensing inventory?
WITH song_moods AS (
  SELECT
    genre,
    ai_classify(
      CONCAT('Song: "', title, '" by ', artist_name, ' (Genre: ', genre, ')'),
      ARRAY('Upbeat', 'Melancholy', 'Energetic', 'Romantic', 'Dark', 'Chill')
    ) AS mood
  FROM songs
)
SELECT
  mood,
  COUNT(*) AS song_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_catalog,
  CONCAT_WS(', ', COLLECT_SET(genre)) AS genres_represented
FROM song_moods
GROUP BY mood
ORDER BY song_count DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC This notebook demonstrated how to use AI Functions in SQL for:
-- MAGIC
-- MAGIC 1. **Mood Classification** - Tag songs by mood for sync licensing (Upbeat, Melancholy, Romantic, etc.)
-- MAGIC 2. **Entity Extraction** - Pull structured data from unstructured license documents
-- MAGIC 3. **Text Generation** - Create executive summaries and personalized notifications
-- MAGIC 4. **Sentiment Analysis** - Triage partner communications by urgency
-- MAGIC 5. **Similarity Search** - Find related songs based on semantic similarity
-- MAGIC
-- MAGIC These AI Functions enable analysts to leverage LLMs directly in SQL without any Python code,
-- MAGIC making advanced AI capabilities accessible to the entire data team.
-- MAGIC
-- MAGIC ### Next Steps
-- MAGIC Proceed to **06_agent_tools.sql** to create the UC functions that will power our Mosaic AI agent.
