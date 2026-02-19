# Databricks notebook source
# MAGIC %md
# MAGIC # GMR Mosaic AI Demo - Build Agent
# MAGIC
# MAGIC **Business Context:** This notebook constructs a Mosaic AI agent that enables GMR analysts
# MAGIC to interact with royalty data using natural language. The agent uses function calling to
# MAGIC invoke Unity Catalog tools and provides accurate, grounded responses.
# MAGIC
# MAGIC ## Agent Architecture
# MAGIC ```
# MAGIC User Query → LLM → Tool Selection → UC Function Execution → Response
# MAGIC                    ↓
# MAGIC              Vector Search (if semantic search needed)
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install mlflow>=2.14 databricks-agents databricks-sdk langchain langchain-community --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "gmr_demo_catalog", "Catalog Name")
dbutils.widgets.text("schema", "royalties", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# Agent configuration
AGENT_NAME = "gmr_royalty_agent"
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
EXPERIMENT_NAME = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/gmr_agent_experiment"

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"LLM Endpoint: {LLM_ENDPOINT}")

# COMMAND ----------

import mlflow
import json

# Set the experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Enable MLflow tracing
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Agent Tools
# MAGIC
# MAGIC We'll register our UC functions as tools that the agent can call.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Define the tools from our UC functions
TOOLS = [
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.lookup_song_royalties",
            "description": "Look up royalty payment history for a song. Use this when the user asks about royalties, payments, or earnings for a specific song. Search by song title or ISRC code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Song ID (e.g. SONG000001), song title, or ISRC code to search for"
                    }
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.search_song_catalog",
            "description": "Search the song catalog using semantic similarity. Use this when the user wants to find songs by description, mood, genre, or any natural language query like 'upbeat pop songs' or 'acoustic ballads about love'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of songs to find"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.calculate_royalty_split",
            "description": "Calculate how a royalty payment should be split among songwriters. Use this when the user wants to know the per-songwriter breakdown for a specific payment amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_song_id": {
                        "type": "string",
                        "description": "The song_id to calculate splits for (e.g. SONG000001)"
                    },
                    "gross_amount": {
                        "type": "number",
                        "description": "The gross royalty amount to distribute"
                    }
                },
                "required": ["input_song_id", "gross_amount"]
            }
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.get_licensing_summary",
            "description": "Get licensing statistics for a song or artist. Use this when the user asks about licensing deals, territories, license types, or licensing revenue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Song title, song_id, or artist name to search for"
                    }
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.lookup_songwriter_earnings",
            "description": "Look up earnings for a songwriter by their ID (e.g. SW00001) or name. Use this when the user asks about a songwriter's earnings, payments, income, or how much money a specific songwriter has made.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Songwriter ID (e.g. SW00001) or songwriter name to search for"
                    }
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.get_top_royalty_songs",
            "description": "Get the top royalty-earning songs ranked by total net payments. Use this when the user asks about top songs, highest earners, best performers, most profitable songs, or royalty rankings. Supports optional genre and period filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_results": {
                        "type": "integer",
                        "description": "Number of top songs to return (default 10, max 50)"
                    },
                    "filter_genre": {
                        "type": "string",
                        "description": "Genre filter (e.g. Pop, Rock, Hip-Hop). MUST pass null for all genres."
                    },
                    "filter_period": {
                        "type": "string",
                        "description": "Payment period filter (e.g. 2025-Q1). MUST pass null for all periods."
                    }
                },
                "required": ["num_results", "filter_genre", "filter_period"]
            }
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.get_catalog_overview",
            "description": "Get a high-level overview of the GMR catalog including total songs, artists, songwriters, revenue totals, and top-performing genre. Use this when the user asks for a portfolio summary, catalog overview, dashboard stats, or general statistics about the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_genre": {
                        "type": "string",
                        "description": "Optional genre filter (e.g. Pop, Rock). MUST pass null for entire catalog overview."
                    }
                },
                "required": ["filter_genre"]
            }
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.get_revenue_by_territory",
            "description": "Get licensing revenue breakdown by geographic territory. Use this when the user asks about revenue by country, territory performance, geographic distribution, international licensing, or market analysis. Optionally filter to a specific territory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_territory": {
                        "type": "string",
                        "description": "Territory code filter (e.g. US, UK, DE, JP). MUST pass null for all territories."
                    }
                },
                "required": ["filter_territory"]
            }
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.get_platform_performance",
            "description": "Get performance metrics by streaming platform, radio station, or venue. Use this when the user asks about platform performance, streaming numbers, which platform pays the most, play counts by platform, or comparisons between Spotify, Apple Music, etc. Optionally filter to a specific platform.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_platform": {
                        "type": "string",
                        "description": "Platform name (e.g. Spotify, Apple Music, iHeartRadio). MUST pass null for all platforms."
                    }
                },
                "required": ["filter_platform"]
            }
        }
    }
]

print(f"Registered {len(TOOLS)} tools")
for tool in TOOLS:
    print(f"  - {tool['uc_function']['name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Agent
# MAGIC
# MAGIC We'll create a Mosaic AI agent using the `databricks-agents` framework.

# COMMAND ----------

from databricks import agents

# Define the agent configuration
agent_config = {
    "llm_endpoint": LLM_ENDPOINT,
    "tools": TOOLS,
    "system_prompt": """You are the GMR Royalty Assistant, an AI agent that helps Global Music Rights analysts
answer questions about songs, royalties, licenses, and songwriters.

You have access to the following tools:
1. lookup_song_royalties - Get payment history for a song
2. search_song_catalog - Find songs using natural language descriptions
3. calculate_royalty_split - Calculate per-songwriter payment breakdown
4. get_licensing_summary - Get licensing statistics
5. lookup_songwriter_earnings - Get earnings summary for a songwriter by ID or name
6. get_top_royalty_songs - Get ranked list of highest-earning songs (supports genre/period filters)
7. get_catalog_overview - Get high-level portfolio stats (total songs, artists, revenue)
8. get_revenue_by_territory - Get licensing revenue breakdown by geographic territory
9. get_platform_performance - Get play counts and royalties by streaming platform/radio station
7. get_catalog_overview - Get high-level portfolio stats (total songs, artists, revenue)
8. get_revenue_by_territory - Get licensing revenue breakdown by country/territory
9. get_platform_performance - Get streaming/radio play counts and platform analytics

Guidelines:
- Always use the appropriate tool to get accurate data before answering
- Provide specific numbers and details from the tool results
- If a query is ambiguous, ask for clarification
- Never make up royalty numbers - always query the actual data
- Format currency values with proper formatting ($X,XXX.XX)
- Be concise but thorough in your responses
- If you can't find the requested information, clearly state that

Remember: Accuracy in royalty reporting is critical for GMR's mission.""",
    "max_iterations": 5,
    "warehouse_id": ""  # Resolved at runtime via DATABRICKS_WAREHOUSE_ID env var in Model Serving
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Agent Class

# COMMAND ----------

from mlflow.pyfunc import PythonModel
from mlflow.models import infer_signature
import mlflow.deployments
from typing import Dict, Any, List, Optional

class GMRRoyaltyAgent(PythonModel):
    """
    GMR Royalty Assistant Agent using Mosaic AI function calling.
    Returns OpenAI chat format for Databricks AI Playground compatibility.
    """

    def __init__(self):
        self.client = None
        self.config = None

    def load_context(self, context):
        """Load model context and initialize clients."""
        import json
        import mlflow.deployments
        self.client = mlflow.deployments.get_deploy_client("databricks")

        # Load config from model artifacts
        with open(context.artifacts["config"], "r") as f:
            self.config = json.load(f)

        # Build tool name mapping: short_name -> full UC function name
        # OpenAI function calling spec requires names matching ^[a-zA-Z0-9_-]{1,64}$
        # so we use the base function name (without catalog.schema prefix)
        self.tool_name_map = {}
        self.openai_tools = []
        for tool in self.config["tools"]:
            if tool["type"] == "uc_function":
                func = tool["uc_function"]
                full_name = func["name"]
                short_name = full_name.split(".")[-1]
                self.tool_name_map[short_name] = full_name
                self.openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": short_name,
                        "description": func["description"],
                        "parameters": func.get("parameters", {"type": "object", "properties": {}})
                    }
                })

    def _call_llm(self, messages: List[Dict], tools: List[Dict] = None) -> Dict:
        """Call the LLM endpoint."""
        payload = {
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1
        }
        if tools:
            payload["tools"] = tools

        response = self.client.predict(
            endpoint=self.config["llm_endpoint"],
            inputs=payload
        )
        return response

    def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute a UC function tool via the Statement Execution API (no PySpark needed)."""
        import json
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.sql import StatementState
        import os

        w = WorkspaceClient()

        # Parse the function name
        parts = tool_name.split(".")
        catalog, schema, function = parts[0], parts[1], parts[2]

        # Build the SQL query with properly escaped arguments
        def escape_sql_value(v):
            if v is None:
                return "NULL"
            if isinstance(v, str):
                # LLMs sometimes send the literal string "NULL" or "null"
                # instead of a JSON null - treat these as SQL NULL
                if v.upper() == "NULL":
                    return "NULL"
                # Escape single quotes to prevent SQL injection
                escaped = v.replace("'", "''")
                return f"'{escaped}'"
            return str(v)

        # Use named parameters (param => value) to avoid argument order issues
        # since the LLM may send arguments in any order
        args_str = ", ".join([f"{k} => {escape_sql_value(v)}" for k, v in arguments.items()])
        query = f"SELECT * FROM {catalog}.{schema}.{function}({args_str})"

        # Use the warehouse ID from environment or config
        warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", self.config.get("warehouse_id", ""))

        # Execute via Statement Execution API (works in Model Serving containers)
        response = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=query,
            wait_timeout="30s",
            catalog=catalog,
            schema=schema
        )

        if response.status.state != StatementState.SUCCEEDED:
            error_msg = response.status.error.message if response.status.error else "Unknown error"
            return json.dumps({"error": error_msg})

        # Convert result to JSON
        columns = [col.name for col in response.manifest.schema.columns]
        rows = []
        if response.result and response.result.data_array:
            for row in response.result.data_array:
                rows.append(dict(zip(columns, row)))

        return json.dumps(rows)

    def _make_chat_response(self, content: str) -> Dict:
        """Return response in OpenAI chat completions format for Playground compatibility."""
        return {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }

    def predict(self, context, model_input, params=None) -> Dict:
        """
        Process a user query through the agent.

        Accepts OpenAI chat format input and returns OpenAI chat format output
        for Databricks AI Playground compatibility.
        """
        import json

        # Handle input: OpenAI chat format sends {"messages": [...]}
        if isinstance(model_input, dict):
            messages = model_input.get("messages", [])
        else:
            # Handle pandas DataFrame from MLflow serving
            import pandas as pd
            if isinstance(model_input, pd.DataFrame):
                row = model_input.iloc[0].to_dict()
                messages = row.get("messages", [])
                if isinstance(messages, str):
                    messages = json.loads(messages)
            else:
                messages = [{"role": "user", "content": str(model_input)}]

        # Add system prompt
        full_messages = [
            {"role": "system", "content": self.config["system_prompt"]}
        ] + messages

        # Iterative tool calling loop
        for iteration in range(self.config.get("max_iterations", 5)):
            response = self._call_llm(full_messages, self.openai_tools)

            # Check if we have a final response or need to call tools
            choice = response["choices"][0]
            message = choice["message"]

            if choice.get("finish_reason") == "tool_calls" or message.get("tool_calls"):
                # Execute each tool call
                tool_calls = message.get("tool_calls", [])
                full_messages.append(message)

                for tool_call in tool_calls:
                    func = tool_call["function"]
                    short_name = func["name"]
                    arguments = json.loads(func.get("arguments", "{}"))

                    # Map short name back to full UC function name
                    full_name = self.tool_name_map.get(short_name, short_name)

                    # Execute the tool
                    try:
                        result = self._execute_tool(full_name, arguments)
                        tool_response = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result
                        }
                    except Exception as e:
                        tool_response = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": f"Error executing tool: {str(e)}"
                        }

                    full_messages.append(tool_response)
            else:
                return self._make_chat_response(message.get("content", ""))

        return self._make_chat_response("Max iterations reached without final response.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the Agent to MLflow

# COMMAND ----------

# Save config to file
import tempfile
import os

with tempfile.TemporaryDirectory() as tmp_dir:
    config_path = os.path.join(tmp_dir, "agent_config.json")
    with open(config_path, "w") as f:
        json.dump(agent_config, f)

    # Declare resource dependencies so the serving container gets proper credentials
    # Including Vector Search index + embedding model so VECTOR_SEARCH() works from serving
    from mlflow.models.resources import (
        DatabricksServingEndpoint,
        DatabricksFunction,
        DatabricksVectorSearchIndex,
    )

    resources = [
        DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
        DatabricksServingEndpoint(endpoint_name="databricks-gte-large-en"),
        DatabricksVectorSearchIndex(index_name=f"{CATALOG}.{SCHEMA}.song_metadata_index"),
    ] + [
        DatabricksFunction(function_name=tool["uc_function"]["name"])
        for tool in TOOLS
    ]

    # Infer chat-compatible signature from examples
    chat_input = {"messages": [{"role": "user", "content": "Hello"}]}
    chat_output = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}]}

    # Log the agent
    with mlflow.start_run(run_name="gmr_royalty_agent_v1") as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="agent",
            python_model=GMRRoyaltyAgent(),
            artifacts={"config": config_path},
            signature=infer_signature(chat_input, chat_output),
            resources=resources,
            pip_requirements=[
                "mlflow>=2.14",
                "databricks-sdk>=0.20.0",
                "pandas",
            ],
            registered_model_name=f"{CATALOG}.{SCHEMA}.{AGENT_NAME}",
            metadata={"task": "llm/v1/chat"}
        )

        # Log parameters and tags
        mlflow.log_params({
            "llm_endpoint": LLM_ENDPOINT,
            "num_tools": len(TOOLS),
            "max_iterations": agent_config["max_iterations"]
        })

        mlflow.set_tags({
            "agent_type": "function_calling",
            "domain": "music_royalties",
            "customer": "GMR"
        })

        run_id = run.info.run_id
        print(f"Agent logged with run_id: {run_id}")
        print(f"Model registered as: {CATALOG}.{SCHEMA}.{AGENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Agent Locally

# COMMAND ----------

# Load the agent for testing
loaded_agent = mlflow.pyfunc.load_model(f"runs:/{run_id}/agent")

# Test query 1: Royalty lookup
test_input = {
    "messages": [
        {"role": "user", "content": "What are the royalty payments for song SONG000001?"}
    ]
}

print("Test 1: Royalty Lookup")
print("-" * 50)
# result = loaded_agent.predict(test_input)
# print(result["response"])
print("(Run this cell interactively to test)")

# COMMAND ----------

# Test query 2: Semantic search
test_input = {
    "messages": [
        {"role": "user", "content": "Find me some upbeat pop songs"}
    ]
}

print("\nTest 2: Semantic Search")
print("-" * 50)
# result = loaded_agent.predict(test_input)
# print(result["response"])
print("(Run this cell interactively to test)")

# COMMAND ----------

# Test query 3: Top earning songs
test_input = {
    "messages": [
        {"role": "user", "content": "What are the top 5 highest-earning songs?"}
    ]
}

print("\nTest 3: Top Earning Songs")
print("-" * 50)
# result = loaded_agent.predict(test_input)
# print(result["response"])
print("(Run this cell interactively to test)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Using Databricks Agents SDK

# COMMAND ----------

# MAGIC %md
# MAGIC For a more streamlined approach, you can use the `databricks-agents` SDK:
# MAGIC
# MAGIC ```python
# MAGIC from databricks import agents
# MAGIC
# MAGIC # Create agent using the SDK
# MAGIC agent = agents.create_agent(
# MAGIC     name="gmr_royalty_agent",
# MAGIC     llm_endpoint="databricks-claude-sonnet-4-5",
# MAGIC     tools=[
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.lookup_song_royalties"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.search_song_catalog"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.calculate_royalty_split"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.get_licensing_summary"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.lookup_songwriter_earnings"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.get_top_royalty_songs"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.get_catalog_overview"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.get_revenue_by_territory"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.get_platform_performance"),
# MAGIC     ],
# MAGIC     system_prompt=agent_config["system_prompt"]
# MAGIC )
# MAGIC
# MAGIC # Register the agent
# MAGIC agents.register_agent(
# MAGIC     agent=agent,
# MAGIC     model_name=f"{CATALOG}.{SCHEMA}.{AGENT_NAME}",
# MAGIC     description="GMR Royalty Assistant for querying song royalties and licensing data"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook created and registered a Mosaic AI agent with:
# MAGIC
# MAGIC - **9 UC Function Tools** for royalty queries, semantic search, songwriter earnings, territory analysis, and platform performance
# MAGIC - **Function Calling** capability using the LLM's tool use feature
# MAGIC - **MLflow Registration** in Unity Catalog for versioning and deployment
# MAGIC - **Tracing Enabled** for debugging and monitoring
# MAGIC
# MAGIC The agent is now registered at: `{CATALOG}.{SCHEMA}.{AGENT_NAME}`
# MAGIC
# MAGIC ### Next Steps
# MAGIC Proceed to **08_evaluate_agent.py** to evaluate the agent using LLM judges.

# COMMAND ----------

print(f"""
Agent Build Complete!
=====================
Model Name: {CATALOG}.{SCHEMA}.{AGENT_NAME}
MLflow Run ID: {run_id}
Experiment: {EXPERIMENT_NAME}

Tools Available:
- lookup_song_royalties
- search_song_catalog
- calculate_royalty_split
- get_licensing_summary
- lookup_songwriter_earnings
- get_top_royalty_songs
- get_catalog_overview
- get_revenue_by_territory
- get_platform_performance

Next: Run 08_evaluate_agent.py to evaluate agent quality
""")
