-- Databricks notebook source
-- MAGIC %md
-- MAGIC # GMR Mosaic AI Demo - Monitoring Dashboard
-- MAGIC
-- MAGIC **Business Context:** This notebook provides SQL queries for monitoring the GMR Royalty Assistant
-- MAGIC in production. These queries can be used to create a Lakeview dashboard for real-time monitoring.
-- MAGIC
-- MAGIC ## Dashboard Sections
-- MAGIC 1. Request Volume & Trends
-- MAGIC 2. Latency Metrics (P50, P95, P99)
-- MAGIC 3. Error Rates & Types
-- MAGIC 4. Token Usage & Costs
-- MAGIC 5. Guardrail Trigger Analysis
-- MAGIC 6. User Activity Patterns

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Configuration
-- MAGIC
-- MAGIC Adjust these variables for your deployment:

-- COMMAND ----------

-- Set catalog and schema
USE CATALOG gmr_demo;
USE SCHEMA royalties;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. Request Volume Overview

-- COMMAND ----------

-- Daily request volume (last 30 days)
-- NOTE: Replace with actual inference table names after deployment

CREATE OR REPLACE TEMPORARY VIEW daily_volume AS
SELECT
  DATE(timestamp) AS date,
  COUNT(*) AS total_requests,
  COUNT(DISTINCT client_request_id) AS unique_sessions,
  ROUND(AVG(total_tokens), 0) AS avg_tokens_per_request
FROM system.serving.served_model_inference_logs  -- Replace with your inference table
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY 1
ORDER BY 1;

-- Display results
SELECT * FROM daily_volume;

-- COMMAND ----------

-- Hourly request pattern (typical day)
SELECT
  HOUR(timestamp) AS hour_of_day,
  DAYOFWEEK(timestamp) AS day_of_week,
  COUNT(*) AS request_count
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY 1, 2
ORDER BY 2, 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Latency Metrics

-- COMMAND ----------

-- Latency percentiles by day
SELECT
  DATE(timestamp) AS date,
  COUNT(*) AS requests,
  ROUND(PERCENTILE(execution_time_ms, 0.50), 0) AS p50_latency_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.90), 0) AS p90_latency_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.95), 0) AS p95_latency_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.99), 0) AS p99_latency_ms,
  ROUND(MAX(execution_time_ms), 0) AS max_latency_ms
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 14 DAYS
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
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
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
  DATE(timestamp) AS date,
  COUNT(*) AS total_requests,
  SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) AS error_count,
  ROUND(100.0 * SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) / COUNT(*), 2) AS error_rate_pct
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 14 DAYS
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- Error breakdown by status code
SELECT
  status_code,
  COUNT(*) AS error_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND status_code != 200
  AND timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY 1
ORDER BY 2 DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4. Token Usage & Cost Estimation

-- COMMAND ----------

-- Daily token usage
SELECT
  DATE(timestamp) AS date,
  SUM(prompt_tokens) AS total_prompt_tokens,
  SUM(completion_tokens) AS total_completion_tokens,
  SUM(total_tokens) AS total_tokens,
  ROUND(AVG(total_tokens), 0) AS avg_tokens_per_request,
  -- Estimated cost (adjust rates based on your model)
  ROUND(SUM(prompt_tokens) * 0.00001 + SUM(completion_tokens) * 0.00003, 2) AS estimated_cost_usd
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- Token usage by request type (based on input pattern)
SELECT
  CASE
    WHEN request LIKE '%royalt%' THEN 'Royalty Lookup'
    WHEN request LIKE '%search%' OR request LIKE '%find%' THEN 'Song Search'
    WHEN request LIKE '%split%' THEN 'Royalty Split'
    WHEN request LIKE '%licens%' THEN 'Licensing'
    WHEN request LIKE '%anomal%' THEN 'Anomaly Detection'
    ELSE 'Other'
  END AS query_type,
  COUNT(*) AS request_count,
  ROUND(AVG(total_tokens), 0) AS avg_tokens,
  SUM(total_tokens) AS total_tokens
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY 1
ORDER BY 4 DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5. Guardrail Trigger Analysis

-- COMMAND ----------

-- Guardrail triggers by type (simulated - replace with actual guardrail logs)
-- This query assumes guardrail events are logged to a separate table

/*
SELECT
  DATE(timestamp) AS date,
  guardrail_type,
  trigger_reason,
  COUNT(*) AS trigger_count
FROM gmr_demo.royalties.agent_guardrail_events
WHERE timestamp >= CURRENT_DATE() - INTERVAL 14 DAYS
GROUP BY 1, 2, 3
ORDER BY 1, 4 DESC;
*/

-- Simulated guardrail analysis based on request patterns
SELECT
  'PII_BLOCKED' AS guardrail_type,
  COUNT(*) AS estimated_triggers
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
  AND (
    LOWER(request) LIKE '%email%'
    OR LOWER(request) LIKE '%ipi%'
    OR LOWER(request) LIKE '%phone%'
  )

UNION ALL

SELECT
  'OFF_TOPIC_BLOCKED' AS guardrail_type,
  COUNT(*) AS estimated_triggers
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
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
  requester AS user_id,
  COUNT(*) AS request_count,
  COUNT(DISTINCT DATE(timestamp)) AS active_days,
  MIN(timestamp) AS first_request,
  MAX(timestamp) AS last_request
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY 1
ORDER BY 2 DESC
LIMIT 20;

-- COMMAND ----------

-- User engagement over time
SELECT
  DATE(timestamp) AS date,
  COUNT(DISTINCT requester) AS unique_users,
  COUNT(*) AS total_requests,
  ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT requester), 1) AS requests_per_user
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 30 DAYS
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 7. Agent Tool Usage

-- COMMAND ----------

-- Tool invocation patterns (based on request analysis)
SELECT
  CASE
    WHEN LOWER(request) LIKE '%royalt%' AND LOWER(request) NOT LIKE '%split%' THEN 'lookup_song_royalties'
    WHEN LOWER(request) LIKE '%search%' OR LOWER(request) LIKE '%find%songs%' THEN 'search_song_catalog'
    WHEN LOWER(request) LIKE '%split%' THEN 'calculate_royalty_split'
    WHEN LOWER(request) LIKE '%licens%' THEN 'get_licensing_summary'
    WHEN LOWER(request) LIKE '%anomal%' THEN 'flag_payment_anomaly'
    ELSE 'unknown'
  END AS likely_tool,
  COUNT(*) AS invocation_count,
  ROUND(AVG(execution_time_ms), 0) AS avg_latency_ms,
  ROUND(AVG(total_tokens), 0) AS avg_tokens
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
GROUP BY 1
ORDER BY 2 DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 8. SLA Compliance

-- COMMAND ----------

-- SLA metrics (assuming 2-second target latency)
SELECT
  DATE(timestamp) AS date,
  COUNT(*) AS total_requests,
  SUM(CASE WHEN execution_time_ms <= 2000 THEN 1 ELSE 0 END) AS within_sla,
  ROUND(100.0 * SUM(CASE WHEN execution_time_ms <= 2000 THEN 1 ELSE 0 END) / COUNT(*), 2) AS sla_compliance_pct,
  SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) AS successful,
  ROUND(100.0 * SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) / COUNT(*), 2) AS success_rate_pct
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 14 DAYS
GROUP BY 1
ORDER BY 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 9. Executive Summary View

-- COMMAND ----------

-- Executive summary for the last 7 days
SELECT
  'Last 7 Days' AS period,
  COUNT(*) AS total_requests,
  COUNT(DISTINCT requester) AS unique_users,
  ROUND(AVG(execution_time_ms), 0) AS avg_latency_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.95), 0) AS p95_latency_ms,
  ROUND(100.0 * SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) / COUNT(*), 2) AS success_rate_pct,
  SUM(total_tokens) AS total_tokens,
  ROUND(SUM(prompt_tokens) * 0.00001 + SUM(completion_tokens) * 0.00003, 2) AS estimated_cost_usd
FROM system.serving.served_model_inference_logs
WHERE served_model_name LIKE '%gmr%'
  AND timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS;

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
-- MAGIC | Section | Visualization Type | Query |
-- MAGIC |---------|-------------------|-------|
-- MAGIC | Header | Counter | Executive Summary |
-- MAGIC | Row 1 | Line Chart | Daily Request Volume |
-- MAGIC | Row 1 | Bar Chart | Latency Percentiles |
-- MAGIC | Row 2 | Line Chart | Error Rates |
-- MAGIC | Row 2 | Pie Chart | Tool Usage Distribution |
-- MAGIC | Row 3 | Heatmap | Hourly Request Pattern |
-- MAGIC | Row 3 | Bar Chart | Top Users |
-- MAGIC | Row 4 | Counter | Guardrail Triggers |
-- MAGIC | Row 4 | Line Chart | SLA Compliance |

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC This notebook provides SQL queries for comprehensive monitoring of the GMR Royalty Assistant:
-- MAGIC
-- MAGIC 1. **Request Metrics** - Volume, trends, patterns
-- MAGIC 2. **Performance** - Latency percentiles, SLA compliance
-- MAGIC 3. **Reliability** - Error rates, success rates
-- MAGIC 4. **Cost** - Token usage, estimated costs
-- MAGIC 5. **Safety** - Guardrail triggers
-- MAGIC 6. **Usage** - User activity, tool usage
-- MAGIC
-- MAGIC ### Next Steps
-- MAGIC
-- MAGIC 1. Create a Lakeview dashboard using these queries
-- MAGIC 2. Set up alerts for:
-- MAGIC    - Error rate > 5%
-- MAGIC    - P95 latency > 5 seconds
-- MAGIC    - Guardrail trigger spikes
-- MAGIC 3. Schedule daily summary reports
-- MAGIC
-- MAGIC ---
-- MAGIC
-- MAGIC **Demo Complete!** You have now built an end-to-end Mosaic AI agent for GMR with:
-- MAGIC - Sample data generation
-- MAGIC - Feature engineering & vector search
-- MAGIC - Agent with UC function tools
-- MAGIC - LLM-as-judge evaluation
-- MAGIC - Production deployment with guardrails
-- MAGIC - Comprehensive monitoring
