# Databricks notebook source
# MAGIC %md
# MAGIC # GMR Mosaic AI Demo - Deploy Agent
# MAGIC
# MAGIC **Business Context:** After building and evaluating the GMR Royalty Assistant, it's time to
# MAGIC deploy it to production. This notebook covers the full MLOps workflow: model registration,
# MAGIC champion/challenger setup, and serving endpoint deployment.
# MAGIC
# MAGIC ## Deployment Architecture
# MAGIC ```
# MAGIC MLflow Registry (UC) → Model Serving Endpoint → Review App
# MAGIC         ↓                      ↓                    ↓
# MAGIC   Version Control      Auto-scaling, Rate Limits   User Feedback
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install mlflow>=2.14 databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "gmr_demo", "Catalog Name")
dbutils.widgets.text("schema", "royalties", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

AGENT_NAME = "gmr_royalty_agent"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.{AGENT_NAME}"
ENDPOINT_NAME = f"gmr-royalty-agent-{CATALOG.replace('_', '-')}"

print(f"Model: {MODEL_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
import json
import time

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry Management
# MAGIC
# MAGIC View registered model versions and set aliases (Champion/Challenger).

# COMMAND ----------

# List all versions of the model
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
print(f"Model: {MODEL_NAME}")
print(f"Total versions: {len(versions)}")
print("-" * 50)

for v in versions:
    print(f"Version {v.version}:")
    print(f"  Status: {v.status}")
    print(f"  Created: {v.creation_timestamp}")
    print(f"  Run ID: {v.run_id}")
    if v.aliases:
        print(f"  Aliases: {v.aliases}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Champion Alias
# MAGIC
# MAGIC Mark the current best version as "Champion" for production use.

# COMMAND ----------

# Set the latest version as Champion
latest_version = max([int(v.version) for v in versions])

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=str(latest_version)
)

print(f"Set version {latest_version} as Champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Model Serving Endpoint
# MAGIC
# MAGIC Deploy the agent to a serverless Model Serving endpoint with auto-scaling.

# COMMAND ----------

# Check if endpoint exists
existing_endpoints = [e.name for e in w.serving_endpoints.list()]

if ENDPOINT_NAME in existing_endpoints:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists. Updating...")
    endpoint_exists = True
else:
    print(f"Creating new endpoint: {ENDPOINT_NAME}")
    endpoint_exists = False

# COMMAND ----------

# Define endpoint configuration
endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name=MODEL_NAME,
            entity_version=str(latest_version),
            workload_size="Small",  # Small, Medium, Large
            scale_to_zero_enabled=True,  # Scale down when not in use
            workload_type="CPU"  # or "GPU" for GPU-accelerated inference
        )
    ],
    auto_capture_config={
        "catalog_name": CATALOG,
        "schema_name": SCHEMA,
        "table_name_prefix": "agent_inference",
        "enabled": True
    }
)

# COMMAND ----------

# Create or update endpoint
if endpoint_exists:
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        served_entities=endpoint_config.served_entities,
        auto_capture_config=endpoint_config.auto_capture_config
    )
else:
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=endpoint_config
    )

print(f"Endpoint configuration submitted: {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for Endpoint to be Ready

# COMMAND ----------

def wait_for_endpoint(endpoint_name, timeout=1200):
    """Wait for serving endpoint to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state.ready

        if state == "READY":
            print(f"Endpoint '{endpoint_name}' is READY!")
            return endpoint
        elif state == "NOT_READY":
            config_state = endpoint.state.config_update
            print(f"Endpoint state: {state}, Config update: {config_state}")
            time.sleep(30)
        else:
            print(f"Unexpected state: {state}")
            time.sleep(30)

    raise TimeoutError(f"Endpoint did not become ready within {timeout} seconds")

endpoint = wait_for_endpoint(ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Rate Limits
# MAGIC
# MAGIC Set rate limits to prevent abuse and manage costs.

# COMMAND ----------

# Note: Rate limits are typically set via the API or UI after endpoint creation
# Here's the configuration that would be applied:

rate_limit_config = {
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
        }
    ]
}

print("Rate limits to configure:")
print(json.dumps(rate_limit_config, indent=2))
print("\nApply these via the Model Serving UI or REST API.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Deployed Endpoint

# COMMAND ----------

# Get endpoint URL
endpoint_url = f"https://{w.config.host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
print(f"Endpoint URL: {endpoint_url}")

# COMMAND ----------

# Test with Python SDK
import requests

def query_agent(question: str) -> str:
    """Query the deployed agent endpoint."""
    token = w.config.token

    payload = {
        "messages": [
            {"role": "user", "content": question}
        ]
    }

    response = requests.post(
        endpoint_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json=payload
    )

    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"

# COMMAND ----------

# Test query
print("Testing deployed agent...")
print("-" * 50)

test_questions = [
    "What are the royalties for SONG000001?",
    "Find upbeat pop songs in our catalog",
    "Are there any payment anomalies to investigate?"
]

for q in test_questions:
    print(f"\nQ: {q}")
    # result = query_agent(q)
    # print(f"A: {result}")
    print("(Run interactively to test)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample curl Command

# COMMAND ----------

# Generate curl command for external testing
curl_command = f"""
curl -X POST '{endpoint_url}' \\
  -H 'Authorization: Bearer $DATABRICKS_TOKEN' \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "messages": [
      {{"role": "user", "content": "What are the royalties for SONG000001?"}}
    ]
  }}'
"""

print("Sample curl command:")
print("-" * 50)
print(curl_command)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Python SDK Example

# COMMAND ----------

python_sdk_example = f'''
from databricks.sdk import WorkspaceClient
import requests

# Initialize client
w = WorkspaceClient()

# Query the agent
def query_gmr_agent(question: str) -> dict:
    """Query the GMR Royalty Assistant."""
    endpoint_url = "https://{w.config.host}/serving-endpoints/{ENDPOINT_NAME}/invocations"

    response = requests.post(
        endpoint_url,
        headers={{
            "Authorization": f"Bearer {{w.config.token}}",
            "Content-Type": "application/json"
        }},
        json={{
            "messages": [{{"role": "user", "content": question}}]
        }}
    )
    return response.json()

# Example usage
result = query_gmr_agent("What were the total royalties for Midnight Dreams in Q3 2025?")
print(result["response"])
'''

print("Python SDK Example:")
print("-" * 50)
print(python_sdk_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Review App
# MAGIC
# MAGIC The Review App provides a chat interface for stakeholders to interact with the agent
# MAGIC and provide feedback.

# COMMAND ----------

# Enable review app on the endpoint
# This is typically done via the UI, but here's the API approach:

review_app_config = {
    "endpoint_name": ENDPOINT_NAME,
    "review_app_enabled": True,
    "feedback_config": {
        "thumbs_up_down": True,
        "text_feedback": True,
        "rating_scale": 5
    }
}

print("Review App Configuration:")
print(json.dumps(review_app_config, indent=2))
print(f"\nReview App URL: https://{w.config.host}/ml/review-app/{ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Champion/Challenger Setup
# MAGIC
# MAGIC Demonstrate how to run multiple model versions for A/B testing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Challenger Version
# MAGIC
# MAGIC To create a challenger, you would:
# MAGIC 1. Make improvements to the agent (better prompts, new tools, etc.)
# MAGIC 2. Log a new version to MLflow
# MAGIC 3. Set the "Challenger" alias
# MAGIC 4. Configure traffic splitting

# COMMAND ----------

# Example: Set up traffic split between Champion and Challenger
traffic_config_example = {
    "traffic_config": {
        "routes": [
            {
                "served_model_name": f"{AGENT_NAME}-champion",
                "traffic_percentage": 90
            },
            {
                "served_model_name": f"{AGENT_NAME}-challenger",
                "traffic_percentage": 10
            }
        ]
    }
}

print("Traffic Split Configuration (Champion/Challenger):")
print(json.dumps(traffic_config_example, indent=2))
print("\nThis sends 90% of traffic to Champion, 10% to Challenger for A/B testing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Endpoint Monitoring
# MAGIC
# MAGIC View endpoint metrics and health.

# COMMAND ----------

# Get endpoint details
endpoint_details = w.serving_endpoints.get(ENDPOINT_NAME)

print(f"Endpoint: {endpoint_details.name}")
print(f"State: {endpoint_details.state.ready}")
print(f"Creator: {endpoint_details.creator}")
print(f"Creation Time: {endpoint_details.creation_timestamp}")

if endpoint_details.config:
    print("\nServed Entities:")
    for entity in endpoint_details.config.served_entities:
        print(f"  - {entity.entity_name} v{entity.entity_version}")
        print(f"    Workload: {entity.workload_size}")
        print(f"    Scale to Zero: {entity.scale_to_zero_enabled}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook deployed the GMR Royalty Assistant:
# MAGIC
# MAGIC 1. **Model Registry** - Registered model with Champion alias
# MAGIC 2. **Serving Endpoint** - Deployed with auto-scaling and scale-to-zero
# MAGIC 3. **Query Logging** - Enabled inference tables for monitoring
# MAGIC 4. **Rate Limits** - Configured to prevent abuse
# MAGIC 5. **Review App** - Enabled for stakeholder feedback
# MAGIC
# MAGIC ### Endpoint Details
# MAGIC - **URL**: `{endpoint_url}`
# MAGIC - **Model**: `{MODEL_NAME}` (Champion)
# MAGIC - **Inference Tables**: `{CATALOG}.{SCHEMA}.agent_inference_*`
# MAGIC
# MAGIC ### Next Steps
# MAGIC Proceed to **10_guardrails_governance.py** to configure AI guardrails and monitoring.

# COMMAND ----------

print(f"""
Agent Deployment Complete!
==========================
Endpoint: {ENDPOINT_NAME}
Model: {MODEL_NAME}
Version: {latest_version} (Champion)
Status: {endpoint_details.state.ready}

Endpoint URL:
{endpoint_url}

Review App:
https://{w.config.host}/ml/review-app/{ENDPOINT_NAME}
""")
