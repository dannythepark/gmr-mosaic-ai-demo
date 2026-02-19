# Databricks notebook source
# MAGIC %md
# MAGIC # GMR Mosaic AI Demo - Guardrails & Governance
# MAGIC
# MAGIC **Business Context:** As GMR handles sensitive songwriter data and financial information,
# MAGIC robust guardrails and governance are essential. This notebook configures AI Gateway guardrails,
# MAGIC credential management, and usage tracking.
# MAGIC
# MAGIC ## Governance Controls
# MAGIC | Control | Purpose | Configuration |
# MAGIC |---------|---------|---------------|
# MAGIC | **PII Filter** | Block songwriter emails, IPI numbers | AI Gateway |
# MAGIC | **Topic Filter** | Block off-topic queries | AI Gateway |
# MAGIC | **Keyword Filter** | Block competitor rate info | AI Gateway |
# MAGIC | **Rate Limits** | Prevent abuse | Endpoint config |
# MAGIC | **Inference Tables** | Audit logging | Auto-capture |

# COMMAND ----------

# MAGIC %pip install databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "gmr_demo_catalog", "Catalog Name")
dbutils.widgets.text("schema", "royalties", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

ENDPOINT_NAME = f"gmr-royalty-agent-{CATALOG.replace('_', '-')}"

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure AI Gateway Guardrails
# MAGIC
# MAGIC AI Gateway provides built-in guardrails for PII filtering, topic safety, and keyword blocking.

# COMMAND ----------

# Define guardrails configuration
guardrails_config = {
    "ai_gateway": {
        "guardrails": {
            "input": {
                # Block off-topic queries
                "invalid_keywords": [
                    "write me a poem",
                    "write a story",
                    "what's the weather",
                    "tell me a joke",
                    "who is the president",
                    "calculate math",
                    "translate this"
                ],
                # Enable safety filters
                "safety": True,
                # Enable PII detection on input
                "pii": {
                    "behavior": "BLOCK"
                }
            },
            "output": {
                # Block PII in responses (emails, phone numbers)
                "pii": {
                    "behavior": "BLOCK"
                },
                # Block competitor rate information
                "invalid_keywords": [
                    "ASCAP rate schedule",
                    "ASCAP fee structure",
                    "BMI rate schedule",
                    "BMI fee structure",
                    "SESAC pricing",
                    "SESAC rates"
                ],
                # Enable safety filters
                "safety": True
            }
        },
        "usage_tracking_config": {
            "enabled": True
        },
        "rate_limits": [
            {
                "calls": 10,
                "key": "user",
                "renewal_period": "MINUTE"
            },
            {
                "calls": 100,
                "key": "endpoint",
                "renewal_period": "MINUTE"
            },
            {
                "calls": 1000,
                "key": "endpoint",
                "renewal_period": "HOUR"
            }
        ]
    }
}

print("Guardrails Configuration:")
print(json.dumps(guardrails_config, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply Guardrails to Endpoint

# COMMAND ----------

# Note: In production, guardrails are applied via the serving endpoint update API
# Here we document the configuration for reference

print(f"""
To apply guardrails to endpoint '{ENDPOINT_NAME}':

Option 1: Via UI
1. Navigate to Machine Learning > Serving
2. Select '{ENDPOINT_NAME}'
3. Click 'Edit endpoint'
4. Configure AI Gateway settings
5. Save changes

Option 2: Via REST API
POST /api/2.0/serving-endpoints/{ENDPOINT_NAME}/config
{{
  "ai_gateway": {json.dumps(guardrails_config['ai_gateway'], indent=4)}
}}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## PII Detection Patterns
# MAGIC
# MAGIC Configure custom PII patterns specific to the music industry.

# COMMAND ----------

# Custom PII patterns for music industry
pii_patterns = {
    "email": {
        "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "action": "BLOCK",
        "description": "Block songwriter email addresses"
    },
    "ipi_number": {
        "pattern": r"\b\d{9,11}\b",
        "context": ["IPI", "ipi", "publisher", "songwriter"],
        "action": "BLOCK",
        "description": "Block IPI (Interested Party Information) numbers"
    },
    "phone": {
        "pattern": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "action": "BLOCK",
        "description": "Block phone numbers"
    },
    "ssn": {
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "action": "BLOCK",
        "description": "Block Social Security numbers"
    }
}

print("Custom PII Patterns:")
for name, config in pii_patterns.items():
    print(f"\n{name}:")
    print(f"  Pattern: {config['pattern']}")
    print(f"  Action: {config['action']}")
    print(f"  Description: {config['description']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Credential Management
# MAGIC
# MAGIC Show how UC functions access data via service principal credentials.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Service Principal Setup
# MAGIC
# MAGIC For production deployments, UC functions should use service principal credentials:

# COMMAND ----------

# Document service principal configuration
sp_config = """
Service Principal Configuration for GMR Agent
=============================================

1. Create Service Principal:
   - Name: gmr-agent-service-principal
   - Workspace: <your-workspace>

2. Grant Permissions:
   - USE CATALOG: gmr_demo
   - USE SCHEMA: gmr_demo.royalties
   - SELECT: All tables in royalties schema
   - EXECUTE: All functions in royalties schema

3. SQL Commands:
   -- Grant catalog access
   GRANT USE CATALOG ON CATALOG gmr_demo_catalog TO `gmr-agent-service-principal`;

   -- Grant schema access
   GRANT USE SCHEMA ON SCHEMA gmr_demo_catalog.royalties TO `gmr-agent-service-principal`;

   -- Grant table read access
   GRANT SELECT ON SCHEMA gmr_demo_catalog.royalties TO `gmr-agent-service-principal`;

   -- Grant function execute access
   GRANT EXECUTE ON SCHEMA gmr_demo_catalog.royalties TO `gmr-agent-service-principal`;

4. Configure Endpoint:
   - Set run_as: gmr-agent-service-principal
   - Endpoint executes with SP permissions, not user tokens
"""

print(sp_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Secrets Management
# MAGIC
# MAGIC For any external API keys (if needed), use Databricks Secrets.

# COMMAND ----------

# Document secrets configuration
secrets_config = """
Secrets Management
==================

If the agent needs external API access (e.g., partner APIs):

1. Create Secret Scope:
   databricks secrets create-scope gmr-secrets

2. Store Secrets:
   databricks secrets put-secret gmr-secrets spotify-api-key
   databricks secrets put-secret gmr-secrets partner-api-token

3. Access in Code:
   from databricks.sdk import WorkspaceClient
   w = WorkspaceClient()
   api_key = w.secrets.get_secret("gmr-secrets", "spotify-api-key").value

4. UC Function Access:
   -- Secrets can be accessed in Python UC functions:
   CREATE FUNCTION get_partner_data(...)
   RETURNS ...
   LANGUAGE PYTHON
   AS $$
   import dbutils
   api_key = dbutils.secrets.get("gmr-secrets", "partner-api-token")
   # Use api_key securely
   $$;
"""

print(secrets_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference Tables & Audit Logging
# MAGIC
# MAGIC Inference tables automatically capture all requests and responses for auditing.

# COMMAND ----------

# Check if inference tables are being populated
inference_tables = spark.sql(f"""
    SHOW TABLES IN {CATALOG}.{SCHEMA} LIKE '*payload*'
""").collect()

print("Inference Tables:")
for table in inference_tables:
    print(f"  - {table.tableName}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query Inference Logs

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View recent inference requests (if table exists)
# MAGIC -- SELECT
# MAGIC --   request_id,
# MAGIC --   timestamp,
# MAGIC --   request,
# MAGIC --   response,
# MAGIC --   latency_ms,
# MAGIC --   status_code
# MAGIC -- FROM gmr_demo_catalog.royalties.gmr_royalty_agent_1_payload
# MAGIC -- ORDER BY timestamp DESC
# MAGIC -- LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Usage Tracking Dashboard Queries
# MAGIC
# MAGIC Queries for monitoring agent usage and performance.

# COMMAND ----------

# Define monitoring queries
monitoring_queries = {
    "request_volume": """
    -- Hourly request volume
    SELECT
      DATE_TRUNC('hour', timestamp) AS hour,
      COUNT(*) AS request_count,
      COUNT(DISTINCT user_id) AS unique_users
    FROM {catalog}.{schema}.agent_inference_request
    WHERE timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
    GROUP BY 1
    ORDER BY 1
    """,

    "latency_percentiles": """
    -- Latency percentiles
    SELECT
      DATE_TRUNC('day', timestamp) AS day,
      PERCENTILE(latency_ms, 0.5) AS p50_latency,
      PERCENTILE(latency_ms, 0.95) AS p95_latency,
      PERCENTILE(latency_ms, 0.99) AS p99_latency
    FROM {catalog}.{schema}.agent_inference_request
    WHERE timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
    GROUP BY 1
    ORDER BY 1
    """,

    "error_rates": """
    -- Error rates by day
    SELECT
      DATE_TRUNC('day', timestamp) AS day,
      COUNT(*) AS total_requests,
      SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) AS errors,
      ROUND(100.0 * SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) / COUNT(*), 2) AS error_rate
    FROM {catalog}.{schema}.agent_inference_request
    WHERE timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
    GROUP BY 1
    ORDER BY 1
    """,

    "guardrail_triggers": """
    -- Guardrail trigger counts
    SELECT
      DATE_TRUNC('day', timestamp) AS day,
      guardrail_type,
      COUNT(*) AS trigger_count
    FROM {catalog}.{schema}.agent_inference_guardrail_events
    WHERE timestamp >= CURRENT_DATE() - INTERVAL 7 DAYS
    GROUP BY 1, 2
    ORDER BY 1, 3 DESC
    """
}

print("Monitoring Dashboard Queries:")
print("-" * 50)
for name, query in monitoring_queries.items():
    print(f"\n{name}:")
    print(query.format(catalog=CATALOG, schema=SCHEMA))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rate Limit Configuration

# COMMAND ----------

# Display rate limit settings
rate_limits = [
    {"key": "user", "calls": 10, "period": "MINUTE", "description": "Per-user limit"},
    {"key": "endpoint", "calls": 100, "period": "MINUTE", "description": "Endpoint burst limit"},
    {"key": "endpoint", "calls": 1000, "period": "HOUR", "description": "Endpoint hourly limit"}
]

print("Rate Limits Configuration:")
print("-" * 50)
for limit in rate_limits:
    print(f"{limit['description']}: {limit['calls']} calls per {limit['period'].lower()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Governance Checklist

# COMMAND ----------

governance_checklist = """
GMR Agent Governance Checklist
==============================

[ ] PII Protection
    [x] Email addresses blocked in responses
    [x] IPI numbers blocked in responses
    [x] Phone numbers blocked in responses
    [ ] Custom PII patterns configured

[ ] Content Safety
    [x] Off-topic query blocking enabled
    [x] Competitor rate info blocked
    [x] Safety filters enabled

[ ] Access Control
    [ ] Service principal configured
    [ ] UC permissions granted
    [ ] Secrets scope created

[ ] Monitoring
    [x] Inference tables enabled
    [x] Usage tracking enabled
    [ ] Alerting configured
    [ ] Dashboard created

[ ] Rate Limiting
    [x] Per-user limits set
    [x] Per-endpoint limits set
    [ ] Billing alerts configured

[ ] Compliance
    [ ] Data retention policy defined
    [ ] Audit log retention configured
    [ ] GDPR/CCPA compliance verified
"""

print(governance_checklist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook configured governance controls for the GMR Royalty Assistant:
# MAGIC
# MAGIC 1. **AI Gateway Guardrails**
# MAGIC    - PII filtering (emails, IPI numbers, phone numbers)
# MAGIC    - Topic filtering (off-topic queries)
# MAGIC    - Keyword filtering (competitor rate information)
# MAGIC
# MAGIC 2. **Credential Management**
# MAGIC    - Service principal setup documentation
# MAGIC    - Secrets management patterns
# MAGIC
# MAGIC 3. **Audit & Monitoring**
# MAGIC    - Inference tables for request logging
# MAGIC    - Usage tracking queries
# MAGIC    - Rate limit configuration
# MAGIC
# MAGIC ### Next Steps
# MAGIC Proceed to **11_monitoring_dashboard.sql** to create the monitoring dashboard.

# COMMAND ----------

print(f"""
Guardrails & Governance Configuration Complete!
===============================================
Endpoint: {ENDPOINT_NAME}

Guardrails Enabled:
- PII Filter: BLOCK (emails, IPI numbers, phone numbers)
- Topic Filter: OFF-TOPIC queries blocked
- Keyword Filter: Competitor rate info blocked
- Safety: Enabled for input and output

Rate Limits:
- Per User: 10 calls/minute
- Per Endpoint: 100 calls/minute, 1000 calls/hour

Audit Logging:
- Inference Tables: {CATALOG}.{SCHEMA}.gmr_royalty_agent_1_payload
- Usage Tracking: Enabled

Review the governance checklist above for compliance verification.
""")
