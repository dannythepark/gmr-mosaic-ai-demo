-- Databricks notebook source
-- MAGIC %md
-- MAGIC # GMR Mosaic AI Demo - Monitoring Dashboard
-- MAGIC
-- MAGIC **Business Context:** This notebook provides SQL queries for monitoring the GMR Royalty Assistant
-- MAGIC in production. These queries can be used to create a Lakeview dashboard for real-time monitoring.
-- MAGIC
-- MAGIC ## Data Sources
-- MAGIC | Table | Purpose |
-- MAGIC |-------|---------|
-- MAGIC | `system.serving.endpoint_usage` | Request volume, tokens, status codes |
-- MAGIC | `system.serving.served_entities` | Endpoint/entity metadata (dimension table) |
-- MAGIC | `gmr_demo_catalog.royalties.gmr_royalty_agent_1_payload` | Request/response content, latency |
-- MAGIC
-- MAGIC ## Dashboard Sections
-- MAGIC 1. Request Volume & Trends
-- MAGIC 2. Latency Metrics (P50, P95, P99)
-- MAGIC 3. Error Rates & Types
-- MAGIC 4. Token Usage & Cost Estimation
-- MAGIC 5. Guardrail Trigger Analysis
-- MAGIC 6. User Activity Patterns
-- MAGIC 7. Agent Tool Usage

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Configuration

-- COMMAND ----------

USE CATALOG gmr_demo_catalog;
USE SCHEMA royalties;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Identify GMR Agent Served Entities

-- COMMAND ----------

-- List served entities for our GMR agent endpoint
SELECT
  served_entity_id,
  endpoint_name,
  entity_name,
  entity_version,
  entity_type,
  change_time
FROM system.serving.served_entities
WHERE endpoint_name = 'agents_gmr_demo_catalog-royalties-gmr_royalty_agent'
ORDER BY change_time DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. Request Volume Overview

-- COMMAND ----------

-- Daily request volume (last 30 days)
SELECT
  DATE(u.request_time) AS date,
  COUNT(*) AS total_requests,
  COUNT(DISTINCT u.client_request_id) AS unique_sessions,
  COUNT(DISTINCT u.requester) AS unique_users,
  SUM(CASE WHEN u.status_code = 200 THEN 1 ELSE 0 END) AS successful_requests
FROM system.serving.endpoint_usage u
JOIN system.serving.served_entities e
  ON u.served_entity_id = e.served_entity_id
WHERE e.endpoint_name = 'agents_gmr_demo_catalog-royalties-gmr_royalty_agent'
  AND u.request_time >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- Hourly request pattern (typical week)
SELECT
  HOUR(u.request_time) AS hour_of_day,
  DAYOFWEEK(u.request_time) AS day_of_week,
  COUNT(*) AS request_count
FROM system.serving.endpoint_usage u
JOIN system.serving.served_entities e
  ON u.served_entity_id = e.served_entity_id
WHERE e.endpoint_name = 'agents_gmr_demo_catalog-royalties-gmr_royalty_agent'
  AND u.request_time >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY 1, 2
ORDER BY 2, 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Latency Metrics
-- MAGIC
-- MAGIC Latency data comes from the inference payload table.

-- COMMAND ----------

-- Latency percentiles by day
SELECT
  DATE(TIMESTAMP_MILLIS(timestamp_ms)) AS date,
  COUNT(*) AS requests,
  ROUND(PERCENTILE(execution_time_ms, 0.50), 0) AS p50_latency_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.90), 0) AS p90_latency_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.95), 0) AS p95_latency_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.99), 0) AS p99_latency_ms,
  ROUND(MAX(execution_time_ms), 0) AS max_latency_ms
FROM gmr_demo_catalog.royalties.gmr_royalty_agent_1_payload
WHERE timestamp_ms >= UNIX_MILLIS(CURRENT_TIMESTAMP() - INTERVAL 14 DAYS)
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- Latency distribution buckets
SELECT
  CASE
    WHEN execution_time_ms < 500 THEN '< 500ms'
    WHEN execution_time_ms < 1000 THEN '500ms - 1s'
    WHEN execution_time_ms < 2000 THEN '1s - 2s'
    WHEN execution_time_ms < 5000 THEN '2s - 5s'
    ELSE '> 5s'
  END AS latency_bucket,
  COUNT(*) AS request_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM gmr_demo_catalog.royalties.gmr_royalty_agent_1_payload
WHERE timestamp_ms >= UNIX_MILLIS(CURRENT_TIMESTAMP() - INTERVAL 7 DAYS)
GROUP BY 1
ORDER BY
  CASE latency_bucket
    WHEN '< 500ms' THEN 1
    WHEN '500ms - 1s' THEN 2
    WHEN '1s - 2s' THEN 3
    WHEN '2s - 5s' THEN 4
    ELSE 5
  END;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3. Error Analysis

-- COMMAND ----------

-- Error rates by day
SELECT
  DATE(u.request_time) AS date,
  COUNT(*) AS total_requests,
  SUM(CASE WHEN u.status_code != 200 THEN 1 ELSE 0 END) AS error_count,
  ROUND(100.0 * SUM(CASE WHEN u.status_code != 200 THEN 1 ELSE 0 END) / COUNT(*), 2) AS error_rate_pct
FROM system.serving.endpoint_usage u
JOIN system.serving.served_entities e
  ON u.served_entity_id = e.served_entity_id
WHERE e.endpoint_name = 'agents_gmr_demo_catalog-royalties-gmr_royalty_agent'
  AND u.request_time >= CURRENT_DATE() - INTERVAL 14 DAYS
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- Error breakdown by status code
SELECT
  u.status_code,
  COUNT(*) AS error_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM system.serving.endpoint_usage u
JOIN system.serving.served_entities e
  ON u.served_entity_id = e.served_entity_id
WHERE e.endpoint_name = 'agents_gmr_demo_catalog-royalties-gmr_royalty_agent'
  AND u.status_code != 200
  AND u.request_time >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY 1
ORDER BY 2 DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4. Token Usage & Cost Estimation

-- COMMAND ----------

-- Daily token usage
SELECT
  DATE(u.request_time) AS date,
  SUM(u.input_token_count) AS total_input_tokens,
  SUM(u.output_token_count) AS total_output_tokens,
  SUM(u.input_token_count + u.output_token_count) AS total_tokens,
  ROUND(AVG(u.input_token_count + u.output_token_count), 0) AS avg_tokens_per_request,
  -- Estimated cost (Claude Sonnet 4.5 rates: $3/MTok input, $15/MTok output)
  ROUND(SUM(u.input_token_count) * 0.000003 + SUM(u.output_token_count) * 0.000015, 4) AS estimated_cost_usd
FROM system.serving.endpoint_usage u
JOIN system.serving.served_entities e
  ON u.served_entity_id = e.served_entity_id
WHERE e.endpoint_name = 'agents_gmr_demo_catalog-royalties-gmr_royalty_agent'
  AND u.request_time >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- Token usage by request type (from payload table)
SELECT
  CASE
    WHEN LOWER(request) LIKE '%royalt%' AND LOWER(request) NOT LIKE '%split%' THEN 'Royalty Lookup'
    WHEN LOWER(request) LIKE '%search%' OR LOWER(request) LIKE '%find%' THEN 'Song Search'
    WHEN LOWER(request) LIKE '%split%' THEN 'Royalty Split'
    WHEN LOWER(request) LIKE '%licens%' THEN 'Licensing'
    WHEN LOWER(request) LIKE '%overview%' OR LOWER(request) LIKE '%catalog%' THEN 'Catalog Overview'
    WHEN LOWER(request) LIKE '%territory%' THEN 'Territory Revenue'
    WHEN LOWER(request) LIKE '%platform%' OR LOWER(request) LIKE '%streaming%' THEN 'Platform Performance'
    WHEN LOWER(request) LIKE '%earn%' OR LOWER(request) LIKE '%songwriter%' THEN 'Songwriter Earnings'
    WHEN LOWER(request) LIKE '%top%' AND LOWER(request) LIKE '%song%' THEN 'Top Songs'
    ELSE 'Other'
  END AS query_type,
  COUNT(*) AS request_count,
  ROUND(AVG(execution_time_ms), 0) AS avg_latency_ms
FROM gmr_demo_catalog.royalties.gmr_royalty_agent_1_payload
WHERE timestamp_ms >= UNIX_MILLIS(CURRENT_TIMESTAMP() - INTERVAL 7 DAYS)
GROUP BY 1
ORDER BY 2 DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5. Guardrail Trigger Analysis

-- COMMAND ----------

-- Estimated guardrail triggers based on request content (from payload table)
SELECT
  'PII_REQUEST' AS guardrail_type,
  COUNT(*) AS estimated_triggers
FROM gmr_demo_catalog.royalties.gmr_royalty_agent_1_payload
WHERE timestamp_ms >= UNIX_MILLIS(CURRENT_TIMESTAMP() - INTERVAL 7 DAYS)
  AND (
    LOWER(request) LIKE '%email%'
    OR LOWER(request) LIKE '%ipi%'
    OR LOWER(request) LIKE '%phone%'
  )

UNION ALL

SELECT
  'OFF_TOPIC_REQUEST' AS guardrail_type,
  COUNT(*) AS estimated_triggers
FROM gmr_demo_catalog.royalties.gmr_royalty_agent_1_payload
WHERE timestamp_ms >= UNIX_MILLIS(CURRENT_TIMESTAMP() - INTERVAL 7 DAYS)
  AND (
    LOWER(request) LIKE '%weather%'
    OR LOWER(request) LIKE '%poem%'
    OR LOWER(request) LIKE '%joke%'
  );

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 6. User Activity Patterns

-- COMMAND ----------

-- Top users by request volume
SELECT
  u.requester AS user_id,
  COUNT(*) AS request_count,
  COUNT(DISTINCT DATE(u.request_time)) AS active_days,
  MIN(u.request_time) AS first_request,
  MAX(u.request_time) AS last_request,
  SUM(u.input_token_count + u.output_token_count) AS total_tokens
FROM system.serving.endpoint_usage u
JOIN system.serving.served_entities e
  ON u.served_entity_id = e.served_entity_id
WHERE e.endpoint_name = 'agents_gmr_demo_catalog-royalties-gmr_royalty_agent'
  AND u.request_time >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY 1
ORDER BY 2 DESC
LIMIT 20;

-- COMMAND ----------

-- User engagement over time
SELECT
  DATE(u.request_time) AS date,
  COUNT(DISTINCT u.requester) AS unique_users,
  COUNT(*) AS total_requests,
  ROUND(COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT u.requester), 0), 1) AS requests_per_user
FROM system.serving.endpoint_usage u
JOIN system.serving.served_entities e
  ON u.served_entity_id = e.served_entity_id
WHERE e.endpoint_name = 'agents_gmr_demo_catalog-royalties-gmr_royalty_agent'
  AND u.request_time >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 7. Agent Tool Usage

-- COMMAND ----------

-- Tool invocation patterns (from payload table request content)
SELECT
  CASE
    WHEN LOWER(request) LIKE '%royalt%' AND LOWER(request) NOT LIKE '%split%' THEN 'lookup_song_royalties'
    WHEN LOWER(request) LIKE '%search%' OR LOWER(request) LIKE '%find%songs%' THEN 'search_song_catalog'
    WHEN LOWER(request) LIKE '%split%' THEN 'calculate_royalty_split'
    WHEN LOWER(request) LIKE '%licens%' THEN 'get_licensing_summary'
    WHEN LOWER(request) LIKE '%overview%' OR LOWER(request) LIKE '%catalog%' THEN 'get_catalog_overview'
    WHEN LOWER(request) LIKE '%territory%' THEN 'get_revenue_by_territory'
    WHEN LOWER(request) LIKE '%platform%' OR LOWER(request) LIKE '%streaming%' THEN 'get_platform_performance'
    WHEN LOWER(request) LIKE '%earn%' OR LOWER(request) LIKE '%songwriter%' THEN 'lookup_songwriter_earnings'
    WHEN LOWER(request) LIKE '%top%' AND LOWER(request) LIKE '%song%' THEN 'get_top_royalty_songs'
    ELSE 'unknown'
  END AS likely_tool,
  COUNT(*) AS invocation_count,
  ROUND(AVG(execution_time_ms), 0) AS avg_latency_ms
FROM gmr_demo_catalog.royalties.gmr_royalty_agent_1_payload
WHERE timestamp_ms >= UNIX_MILLIS(CURRENT_TIMESTAMP() - INTERVAL 7 DAYS)
GROUP BY 1
ORDER BY 2 DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 8. SLA Compliance

-- COMMAND ----------

-- SLA metrics (2-second target latency, from payload table for latency + endpoint_usage for success rate)
SELECT
  DATE(u.request_time) AS date,
  COUNT(*) AS total_requests,
  SUM(CASE WHEN u.status_code = 200 THEN 1 ELSE 0 END) AS successful,
  ROUND(100.0 * SUM(CASE WHEN u.status_code = 200 THEN 1 ELSE 0 END) / COUNT(*), 2) AS success_rate_pct
FROM system.serving.endpoint_usage u
JOIN system.serving.served_entities e
  ON u.served_entity_id = e.served_entity_id
WHERE e.endpoint_name = 'agents_gmr_demo_catalog-royalties-gmr_royalty_agent'
  AND u.request_time >= CURRENT_DATE() - INTERVAL 14 DAYS
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- Latency SLA compliance (from payload table)
SELECT
  DATE(TIMESTAMP_MILLIS(timestamp_ms)) AS date,
  COUNT(*) AS total_requests,
  SUM(CASE WHEN execution_time_ms <= 2000 THEN 1 ELSE 0 END) AS within_sla,
  ROUND(100.0 * SUM(CASE WHEN execution_time_ms <= 2000 THEN 1 ELSE 0 END) / COUNT(*), 2) AS sla_compliance_pct
FROM gmr_demo_catalog.royalties.gmr_royalty_agent_1_payload
WHERE timestamp_ms >= UNIX_MILLIS(CURRENT_TIMESTAMP() - INTERVAL 14 DAYS)
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 9. Executive Summary

-- COMMAND ----------

-- Executive summary for the last 7 days
SELECT
  'Last 7 Days' AS period,
  COUNT(*) AS total_requests,
  COUNT(DISTINCT u.requester) AS unique_users,
  ROUND(100.0 * SUM(CASE WHEN u.status_code = 200 THEN 1 ELSE 0 END) / COUNT(*), 2) AS success_rate_pct,
  SUM(u.input_token_count) AS total_input_tokens,
  SUM(u.output_token_count) AS total_output_tokens,
  SUM(u.input_token_count + u.output_token_count) AS total_tokens,
  ROUND(SUM(u.input_token_count) * 0.000003 + SUM(u.output_token_count) * 0.000015, 4) AS estimated_cost_usd
FROM system.serving.endpoint_usage u
JOIN system.serving.served_entities e
  ON u.served_entity_id = e.served_entity_id
WHERE e.endpoint_name = 'agents_gmr_demo_catalog-royalties-gmr_royalty_agent'
  AND u.request_time >= CURRENT_DATE() - INTERVAL 7 DAYS;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Creating a Lakeview Dashboard
-- MAGIC
-- MAGIC To create a Lakeview dashboard from these queries:
-- MAGIC
-- MAGIC 1. Navigate to **SQL Editor** > **Dashboards**
-- MAGIC 2. Click **Create Dashboard**
-- MAGIC 3. Add visualizations using the queries above
-- MAGIC
-- MAGIC ### Recommended Dashboard Layout
-- MAGIC
-- MAGIC | Section | Visualization Type | Data Source |
-- MAGIC |---------|-------------------|------------|
-- MAGIC | Header | Counter | Executive Summary (endpoint_usage) |
-- MAGIC | Row 1 | Line Chart | Daily Request Volume (endpoint_usage) |
-- MAGIC | Row 1 | Bar Chart | Latency Percentiles (payload table) |
-- MAGIC | Row 2 | Line Chart | Error Rates (endpoint_usage) |
-- MAGIC | Row 2 | Pie Chart | Tool Usage Distribution (payload table) |
-- MAGIC | Row 3 | Heatmap | Hourly Request Pattern (endpoint_usage) |
-- MAGIC | Row 3 | Bar Chart | Top Users (endpoint_usage) |
-- MAGIC | Row 4 | Counter | Guardrail Triggers (payload table) |
-- MAGIC | Row 4 | Line Chart | SLA Compliance (both tables) |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC This notebook provides monitoring queries using three data sources:
-- MAGIC
-- MAGIC | Source | Used For |
-- MAGIC |--------|----------|
-- MAGIC | `system.serving.endpoint_usage` | Volume, tokens, errors, users, cost |
-- MAGIC | `system.serving.served_entities` | Endpoint metadata (join dimension) |
-- MAGIC | `gmr_demo_catalog.royalties.gmr_royalty_agent_1_payload` | Latency, request content, tool usage |
-- MAGIC
-- MAGIC ### Next Steps
-- MAGIC 1. Create a Lakeview dashboard using these queries
-- MAGIC 2. Set up alerts for error rate > 5%, P95 latency > 5s
-- MAGIC 3. Schedule daily summary reports
