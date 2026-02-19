# GMR Mosaic AI Agent Demo

An end-to-end demonstration of the Databricks Mosaic AI agent development lifecycle, built for **Global Music Rights (GMR)** - a performance rights organization managing licensing and royalty distribution for music catalogs.

## Overview

This demo showcases how to build, evaluate, deploy, and govern an AI agent using Databricks' unified platform. The agent helps GMR royalty analysts query song data, track payments, and identify anomalies using natural language.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GMR Royalty Assistant                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User Query ──► LLM ──► Tool Selection ──► Response                    │
│                        │                     │                          │
│                        ▼                     ▼                          │
│              ┌─────────────────┐   ┌──────────────────┐                │
│              │ Vector Search   │   │ UC Functions     │                │
│              │ (Semantic)      │   │ (SQL/Python)     │                │
│              └─────────────────┘   └──────────────────┘                │
│                        │                     │                          │
│                        ▼                     ▼                          │
│              ┌─────────────────────────────────────────┐               │
│              │           Unity Catalog                 │               │
│              │  Tables │ Functions │ Models │ Indexes  │               │
│              └─────────────────────────────────────────┘               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Demo Pillars

### 1. Data Preparation
- **Sample Data Generation** - Realistic music rights data (songs, licenses, royalties)
- **Auto Loader Ingestion** - Streaming CSV ingestion with schema evolution
- **Feature Engineering** - ML-ready features (popularity, territory concentration)
- **Vector Search Index** - Semantic song discovery

### 2. Build Agent
- **UC Function Tools** - SQL/Python functions for data queries
- **Foundation Model** - LLM backbone (Claude Sonnet 4.5)
- **Function Calling** - Intelligent tool selection based on user intent
- **AI Functions** - SQL-native AI (classify, extract, generate)

### 3. Evaluate Agent
- **LLM-as-Judge** - Automated quality scoring
- **Evaluation Dataset** - 25 curated Q&A pairs
- **MLflow Tracing** - Full execution visibility
- **Unity Catalog Lineage** - End-to-end data flow tracking

### 4. Deploy Agent
- **Model Registry** - Version control with Champion/Challenger
- **Model Serving** - Auto-scaling serverless endpoint
- **Review App** - Stakeholder feedback collection

### 5. Governance
- **AI Guardrails** - PII filtering, topic blocking
- **Rate Limits** - Usage controls
- **Inference Tables** - Audit logging
- **Credential Management** - Service principal setup

## Quick Start

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to Foundation Model APIs (Claude Sonnet 4.5)
- Vector Search endpoint capability
- Serverless SQL Warehouse

### Deployment

1. **Clone or upload this bundle to your workspace**

2. **Configure the bundle**
   ```bash
   # Edit databricks.yml to set your workspace host
   databricks bundle validate
   ```

3. **Deploy the bundle**
   ```bash
   databricks bundle deploy --target dev
   ```

4. **Run the notebooks in sequence**
   - Start with `01_data_generation.py` to create sample data
   - Progress through each notebook sequentially

## Project Structure

```
gmr-mosaic-ai-demo/
├── databricks.yml                    # Bundle configuration
├── README.md                         # This file
├── resources/
│   ├── jobs.yml                      # Workflow definitions
│   ├── model_serving.yml             # Serving endpoint config
│   └── vector_search.yml             # Vector search config
├── src/
│   ├── 01_data_generation.py         # Sample data creation
│   ├── 02_data_ingestion.py          # Auto Loader patterns
│   ├── 03_feature_engineering.py     # Feature Store tables
│   ├── 04_vector_index.py            # Vector search setup
│   ├── 05_ai_functions.sql           # AI Functions showcase
│   ├── 06_agent_tools.sql            # UC function tools
│   ├── 07_build_agent.py             # Agent construction
│   ├── 08_evaluate_agent.py          # LLM-judge evaluation
│   ├── 09_deploy_agent.py            # Deployment & MLOps
│   ├── 10_guardrails_governance.py   # Guardrails setup
│   └── 11_monitoring_dashboard.sql   # Monitoring queries
└── data/
    ├── eval_dataset.json             # Evaluation Q&A pairs
    └── sample_performance_logs.csv   # Sample CSV for ingestion
```

## Data Model

### Tables Created

| Table | Description | Rows |
|-------|-------------|------|
| `songwriters` | Songwriter profiles with PRO affiliations | 200+ |
| `songs` | Song catalog with ISRC codes | 500+ |
| `licenses` | Licensing agreements by type/territory | 2,000+ |
| `performance_logs` | Play/performance tracking | 50,000+ |
| `royalty_payments` | Payment records to songwriters | 5,000+ |

### Agent Tools

| Tool | Description |
|------|-------------|
| `lookup_song_royalties` | Get royalty payment history |
| `search_song_catalog` | Semantic song search (Vector Search) |
| `calculate_royalty_split` | Per-songwriter payment breakdown |
| `get_licensing_summary` | Licensing stats by song/artist |
| `lookup_songwriter_earnings` | Get earnings summary for a songwriter |
| `get_top_royalty_songs` | Ranked list of highest-earning songs |

## Sample Queries

Once deployed, try these queries with the agent:

```
"What are the royalties for Bohemian Rhapsody?"

"Find upbeat pop songs similar to summer anthems"

"What are the top 10 highest-earning songs?"

"Calculate the royalty split for a $10,000 payment on SONG000001"

"Which territories generate the most licensing revenue?"
```

## Configuration

### Environment Variables

Set these in your `databricks.yml` or as widgets:

| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | Unity Catalog name | `gmr_demo` |
| `schema` | Schema name | `royalties` |
| `warehouse_id` | Serverless SQL Warehouse ID | - |

### Model Endpoints

| Endpoint | Purpose |
|----------|---------|
| `databricks-meta-llama-3-3-70b-instruct` | LLM backbone |
| `databricks-gte-large-en` | Embedding model |
| `databricks-claude-sonnet-4` | LLM-as-judge |

## Customization

### Adding New Tools

1. Create a UC function in `06_agent_tools.sql`
2. Add the tool definition to `07_build_agent.py`
3. Add evaluation questions to `data/eval_dataset.json`
4. Re-deploy the agent

### Changing the LLM

Edit `07_build_agent.py`:
```python
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"  # or other Foundation Model API endpoint
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Vector search index not ready | Wait 5-10 minutes after creation |
| UC function permission denied | Grant EXECUTE to user/service principal |
| Model serving 429 errors | Check rate limits, increase quotas |
| Embedding errors | Verify databricks-gte-large-en endpoint is available |

## Resources

- [Mosaic AI Documentation](https://docs.databricks.com/en/generative-ai/index.html)
- [Unity Catalog Functions](https://docs.databricks.com/en/sql/language-manual/sql-ref-functions-udf.html)
- [Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
- [AI Gateway](https://docs.databricks.com/en/generative-ai/ai-gateway/index.html)

## License

This demo is provided for Databricks customer demonstrations. Sample data is synthetic and does not represent actual GMR data.

---

Built with Databricks Mosaic AI
