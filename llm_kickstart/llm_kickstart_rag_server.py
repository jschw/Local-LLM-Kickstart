from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from fastapi.responses import StreamingResponse
from openai import OpenAI
import uvicorn
from pathlib import Path
import json
import os, appdirs
from llm_kickstart_vectorstore import KickstartVectorsearch

"""
Load and parse the app_config.json file into structured variables.
"""
CONFIG_DIR      = Path(appdirs.user_config_dir(appname='LLM_Kickstart'))
app_config_path = CONFIG_DIR / 'app_config.json'

try:
    if  app_config_path.exists():

        with open(app_config_path, "r") as f:
            app_config = json.load(f)

            rag_proxy_serve_port    = app_config["rag-proxy-serve-port"]
            rag_proxy_llm_port      = app_config["rag-proxy-llm-port"]


except Exception as e:
    print(f"Failed to load config file {app_config_path}: {e}")
    quit()

endpoint_base_url = f"http://localhost:{rag_proxy_llm_port}/v1"
# endpoint_base_url = ""

# Document base dir
doc_base_dir = os.path.expanduser("~/LLM_Kickstart_Documents")

documents_dir = Path(doc_base_dir)
documents_dir.mkdir(parents=True, exist_ok=True)


# Configure OpenAI API key
if endpoint_base_url == "":
    client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
    )

else:
    client = OpenAI(
    api_key="dummy",  # not used locally
    base_url=endpoint_base_url  # your proxy endpoint
    )

app = FastAPI(
    title="Open Prompt Proxy",
    description="A drop-in compatible OpenAI API wrapper that logs prompts and forwards requests.",
    version="1.0.0"
)

# Init RAG
rag_provider        = KickstartVectorsearch()
rag_enabled         = False


# ----------------------------
# /v1/models endpoint
# ----------------------------
@app.get("/v1/models")
async def list_models():
    """Return a list of available models (mirrors OpenAI API)."""
    models = client.models.list()

    # OpenAI returns an OpenAIObject, which is not JSON serializable.
    # Use .to_dict() to get a serializable dictionary.
    return models

# ----------------------------
# /v1/disablerag endpoint
# ----------------------------
@app.get("/v1/disablerag")
async def disable_rag():
    global rag_enabled
    """Disables RAG functionality if a vectorestore was already initialized."""
    print("--> RAG system disabled now.")
    rag_enabled = False

    return {"status": "success"}

# ----------------------------
# /v1/ragupdatepdf endpoint
# ----------------------------
@app.post("/v1/ragupdatepdf")
async def rag_update_pdf(request: Request):
    global rag_enabled
    """
    Accepts a JSON body containing 'document_path',
    loads the PDF, and registers it in a RAG index.
    """
    body = await request.json()

    document_path = body.get("document_path")

    # Update RAG
    rag_update_ok = rag_provider.init_vectorstore_pdf(document_path)

    if rag_update_ok:
        rag_enabled = True
        print("--> RAG update successful, RAG system enabled.")
    else:
        rag_enabled = False
        print("--> RAG update failed, RAG system disabled.")
        return {"status": "failed"}

    return {"status": "success"}

# ----------------------------
# /v1/chat/completions endpoint
# ----------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global rag_enabled

    body = await request.json()

    if rag_enabled:
        # --- Inject RAG context before forwarding ---
        # Query Vectorstore
        messages = body.get("messages", [])
        last_message = messages[-1]  # This is a dict: {"role": "...", "content": "..."}
        last_user_message = last_message.get("content", "")

        rag_output = rag_provider.search_knn(last_user_message)

        rag_context = "The following parts of a document or website should be considered when generating responses and/or answers to the users questions:\n"

        num = 1
        for chunk in rag_output:
            rag_context += f"[\n{num}:\n"
            rag_context += chunk
            rag_context += f"\n],\n"
            num += 1

        rag_context += f"All of the parts of a document or website should only be used if it is helpful in answering the user's question.\n"

        injected_message = {
            "role": "user",
            "content": rag_context
        }

        body["messages"].insert(0, injected_message)  # insert at top
        # OR: body["messages"].append(injected_message)
    
    # Forward the modified body to OpenAI
    response = client.chat.completions.create(**body)

    return response


# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(rag_proxy_serve_port))
