import os
import logging
import json
import httpx
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from databricks.sdk import WorkspaceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GMR Royalty Assistant")

# --- Config ---

IS_DATABRICKS_APP = bool(os.environ.get("DATABRICKS_APP_NAME"))
SERVING_ENDPOINT = os.environ.get(
    "SERVING_ENDPOINT",
    "your-agent-serving-endpoint",
)


def get_workspace_client() -> WorkspaceClient:
    if IS_DATABRICKS_APP:
        return WorkspaceClient()
    profile = os.environ.get("DATABRICKS_PROFILE", "gmr-demo")
    return WorkspaceClient(profile=profile)


# --- API ---


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    try:
        w = get_workspace_client()
        host = w.config.host.rstrip("/")
        headers = w.config.authenticate()
        headers["Content-Type"] = "application/json"

        payload = {
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        }

        logger.info(f"Calling endpoint: {host}/serving-endpoints/{SERVING_ENDPOINT}/invocations")

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{host}/serving-endpoints/{SERVING_ENDPOINT}/invocations",
                headers=headers,
                json=payload,
            )

        logger.info(f"Response status: {resp.status_code}")

        if resp.status_code != 200:
            logger.error(f"Endpoint error: {resp.text}")
            return JSONResponse(
                {"content": f"Error from agent: {resp.text}"},
                status_code=resp.status_code,
            )

        data = resp.json()
        # Agent endpoints return OpenAI-compatible format
        content = ""
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")

        return JSONResponse({"content": content})

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(
            {"content": f"Error: {str(e)}"},
            status_code=500,
        )


# --- Static files ---

frontend_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dir):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(frontend_dir, "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(frontend_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dir, "index.html"))
