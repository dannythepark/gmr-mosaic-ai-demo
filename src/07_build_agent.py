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
# MAGIC User Query → LLM (Claude/Llama) → Tool Selection → UC Function Execution → Response
# MAGIC                    ↓
# MAGIC              Vector Search (if semantic search needed)
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install mlflow>=2.14 databricks-agents databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "gmr_demo", "Catalog Name")
dbutils.widgets.text("schema", "royalties", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# Agent configuration
AGENT_NAME = "gmr_royalty_agent"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"  # Or "databricks-claude-sonnet-4"
EXPERIMENT_NAME = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/gmr_agent_experiment"

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"LLM Endpoint: {LLM_ENDPOINT}")

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
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
            "description": "Look up royalty payment history for a song. Use this when the user asks about royalties, payments, or earnings for a specific song. Search by song title or ISRC code."
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.search_song_catalog",
            "description": "Search the song catalog using semantic similarity. Use this when the user wants to find songs by description, mood, genre, or any natural language query like 'upbeat pop songs' or 'acoustic ballads about love'."
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.calculate_royalty_split",
            "description": "Calculate how a royalty payment should be split among songwriters. Use this when the user wants to know the per-songwriter breakdown for a specific payment amount."
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.get_licensing_summary",
            "description": "Get licensing statistics for a song or artist. Use this when the user asks about licensing deals, territories, license types, or licensing revenue."
        }
    },
    {
        "type": "uc_function",
        "uc_function": {
            "name": f"{CATALOG}.{SCHEMA}.flag_payment_anomaly",
            "description": "Detect unusual payment patterns for a song. Use this when the user wants to check if there are anomalies or potential issues with payments for a specific song or licensee."
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
5. flag_payment_anomaly - Detect unusual payment patterns

Guidelines:
- Always use the appropriate tool to get accurate data before answering
- Provide specific numbers and details from the tool results
- If a query is ambiguous, ask for clarification
- Never make up royalty numbers - always query the actual data
- Format currency values with proper formatting ($X,XXX.XX)
- Be concise but thorough in your responses
- If you can't find the requested information, clearly state that

Remember: Accuracy in royalty reporting is critical for GMR's mission.""",
    "max_iterations": 5
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Agent Class

# COMMAND ----------

from mlflow.pyfunc import PythonModel
import mlflow.deployments
from typing import Dict, Any, List

class GMRRoyaltyAgent(PythonModel):
    """
    GMR Royalty Assistant Agent using Mosaic AI function calling.
    """

    def __init__(self):
        self.client = None
        self.config = None

    def load_context(self, context):
        """Load model context and initialize clients."""
        import mlflow.deployments
        self.client = mlflow.deployments.get_deploy_client("databricks")

        # Load config from model artifacts
        with open(context.artifacts["config"], "r") as f:
            self.config = json.load(f)

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
        """Execute a UC function tool."""
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        # Parse the function name
        parts = tool_name.split(".")
        catalog, schema, function = parts[0], parts[1], parts[2]

        # Build the SQL query
        args_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v)
                             for v in arguments.values()])
        query = f"SELECT * FROM {catalog}.{schema}.{function}({args_str})"

        # Execute the query
        result = spark.sql(query).toPandas()
        return result.to_json(orient="records")

    def predict(self, context, model_input: Dict) -> Dict:
        """
        Process a user query through the agent.

        Args:
            model_input: Dict with 'messages' key containing conversation history

        Returns:
            Dict with 'response' containing the agent's answer
        """
        messages = model_input.get("messages", [])

        # Add system prompt
        full_messages = [
            {"role": "system", "content": self.config["system_prompt"]}
        ] + messages

        # Convert tools to OpenAI format
        openai_tools = []
        for tool in self.config["tools"]:
            if tool["type"] == "uc_function":
                func = tool["uc_function"]
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": func["name"],
                        "description": func["description"],
                        "parameters": {"type": "object", "properties": {}}
                    }
                })

        # Iterative tool calling loop
        for iteration in range(self.config.get("max_iterations", 5)):
            response = self._call_llm(full_messages, openai_tools)

            # Check if we have a final response or need to call tools
            choice = response["choices"][0]
            message = choice["message"]

            if choice.get("finish_reason") == "tool_calls" or message.get("tool_calls"):
                # Execute each tool call
                tool_calls = message.get("tool_calls", [])
                full_messages.append(message)

                for tool_call in tool_calls:
                    func = tool_call["function"]
                    tool_name = func["name"]
                    arguments = json.loads(func.get("arguments", "{}"))

                    # Execute the tool
                    try:
                        result = self._execute_tool(tool_name, arguments)
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
                # Final response
                return {"response": message.get("content", "")}

        return {"response": "Max iterations reached without final response."}

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

    # Create example input/output for signature
    example_input = {
        "messages": [
            {"role": "user", "content": "What were the royalties for Midnight Dreams?"}
        ]
    }
    example_output = {
        "response": "Based on the royalty data..."
    }

    # Log the agent
    with mlflow.start_run(run_name="gmr_royalty_agent_v1") as run:
        # Enable autologging for tracing
        mlflow.langchain.autolog()

        # Log the model
        model_info = mlflow.pyfunc.log_model(
            artifact_path="agent",
            python_model=GMRRoyaltyAgent(),
            artifacts={"config": config_path},
            signature=infer_signature(example_input, example_output),
            pip_requirements=[
                "mlflow>=2.14",
                "databricks-sdk",
                "pandas"
            ],
            registered_model_name=f"{CATALOG}.{SCHEMA}.{AGENT_NAME}"
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

# Test query 3: Anomaly detection
test_input = {
    "messages": [
        {"role": "user", "content": "Are there any payment anomalies for SONG000001?"}
    ]
}

print("\nTest 3: Anomaly Detection")
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
# MAGIC     llm_endpoint="databricks-meta-llama-3-3-70b-instruct",
# MAGIC     tools=[
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.lookup_song_royalties"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.search_song_catalog"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.calculate_royalty_split"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.get_licensing_summary"),
# MAGIC         agents.UCFunctionTool(f"{CATALOG}.{SCHEMA}.flag_payment_anomaly"),
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
# MAGIC - **5 UC Function Tools** for royalty queries, semantic search, and anomaly detection
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
- flag_payment_anomaly

Next: Run 08_evaluate_agent.py to evaluate agent quality
""")
