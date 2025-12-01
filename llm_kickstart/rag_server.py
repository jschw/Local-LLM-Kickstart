from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
import uvicorn
from pathlib import Path
import json
import os, appdirs, time
import uuid
from multiprocessing import Process
from utils_rag import KickstartVectorsearch

class LocalRAGServer:

    def __init__(self, termux_paths=False):
        self.process = None

        self.termux = termux_paths

        CONFIG_DIR                  = Path(appdirs.user_config_dir(appname='LLM_Kickstart'))
        self.rag_server_config_path = CONFIG_DIR / 'rag_server_config.json'
        self.rag_server_config      = None
        self.doc_base_dir           = None
        self.website_crawl_depth    = 1
        self.rag_chunk_count        = 4
        self.rag_proxy_serve_port   = 0
        self.llm_server_port        = 0

        self.rag_score_thresh       = 0.5
        self.rag_max_chunks         = 10

        self.load_config()

    def load_config(self):
        """
        Load and parse the rag_server_config.json file into structured variables.
        """
        try:
            if not self.rag_server_config_path.exists():
                # Create llm config file if not existing
                # Template content of the llm_server_config.json
                if self.termux:
                    doc_base_dir_tmp = "~/storage/shared/LLM_Kickstart/Documents"
                else:
                    doc_base_dir_tmp = "~/LLM_Kickstart/Documents"

                tmp_rag_server_config = {
                    "rag-document-base-dir": doc_base_dir_tmp,
                    "website-crawl-depth": "2",
                    "rag-chunk-count": "5",
                    "enable-query-optimization": "False",
                    "rag-proxy-serve-port": "4001",
                    "llm-server-port": "4000"
                    }

                with self.rag_server_config_path.open('w') as f:
                    json.dump(tmp_rag_server_config, f, indent=4)

            with open(self.rag_server_config_path, "r") as f:
                self.rag_server_config = json.load(f)
                self.rag_proxy_serve_port   = self.rag_server_config["rag-proxy-serve-port"]
                self.llm_server_port        = self.rag_server_config["llm-server-port"]
                self.doc_base_dir           = Path(os.path.expanduser(self.rag_server_config["rag-document-base-dir"]))
                self.website_crawl_depth    = int(self.rag_server_config["website-crawl-depth"])
                self.rag_chunk_count        = int(self.rag_server_config["rag-chunk-count"])
                self.enable_query_opt       = json.loads(str(self.rag_server_config["enable-query-optimization"]).lower())

        except Exception as e:
            print(f"Failed to load config file {self.rag_server_config_path}: {e}")
            self.llm_server_config = None
    
    def _run_server(self):
        endpoint_base_url = f"http://localhost:{self.llm_server_port}/v1"

        self.doc_base_dir.mkdir(parents=True, exist_ok=True)

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

        rag_provider    = KickstartVectorsearch()
        rag_enabled     = False
        summary_enabled = False

        current_rag_summary    = ""

        @app.get("/v1/testmessage")
        async def root():
            return {"message": "FastAPI running inside a class and started from main.py"}

        @app.get("/v1/models")
        async def list_models():
            """Return a list of available models (mirrors OpenAI API)."""
            models = client.models.list()
            models = models.model_dump_json()
            model_list = json.loads(models)

            for model in model_list["data"]:
                mod_name = model["id"]
                mod_name = os.path.basename(mod_name)
                model["id"] = mod_name

            # OpenAI returns an OpenAIObject, which is not JSON serializable.
            # Use .to_dict() to get a serializable dictionary.
            return JSONResponse(model_list)

        @app.get("/v1/disablerag")
        async def disable_rag():
            nonlocal rag_enabled
            """Disables RAG functionality if a vectorestore was already initialized."""
            print("--> RAG system disabled now.")
            rag_enabled = False

            return {"status": "success"}
        
        def rag_update_file(document_path):
            # Split paths if more than one
            document_paths_arg = document_path.split(";")

            # Check document exist
            document_paths_exist = []

            for doc in document_paths_arg:
                doc_current = doc
                if not os.path.isfile(doc_current):
                    # Document is not available at absolute path, checking rel. path
                    doc_current = os.path.join(self.doc_base_dir, doc_current)
                    if not os.path.isfile(doc_current):
                        # Document is not available -> return error
                        print(f"--> Document {doc_current} not found.")
                        continue

                document_paths_exist.append(doc_current)

            if len(document_paths_exist) == 0:
                print("--> No existing document found at given path.")
                return False

            # Update RAG
            rag_update_ok = rag_provider.init_vectorstore_pdf(document_paths_exist)
            return rag_update_ok
        
        def rag_update_web(url, deep):
            # Split paths if more than one
            urls = url.split(";")

            # Update RAG
            rag_update_ok = rag_provider.init_vectorstore_web(urls, deep)

            return rag_update_ok

        @app.post("/v1/ragupdatepdf")
        async def rag_update_pdf(request: Request):
            nonlocal rag_enabled
            """
            Accepts a JSON body containing 'document_path',
            loads the PDF, and registers it in a RAG index.
            """
            body = await request.json()

            document_path = body.get("document_path")

            rag_update_ok = rag_update_file(document_path)

            if rag_update_ok:
                rag_enabled = True
                print("--> RAG update successful, RAG system enabled.")
            else:
                rag_enabled = False
                print("--> RAG update failed, RAG system disabled.")
                return {"status": "failed"}

            return {"status": "success"}
        
        @app.post("/v1/ragupdateweb")
        async def rag_update_website(request: Request):
            nonlocal rag_enabled
            """
            Accepts a JSON body containing 'url',
            loads the PDF, and registers it in a RAG index.
            """
            body = await request.json()

            target_url = body.get("url")

            rag_update_ok = rag_update_web(target_url)

            if rag_update_ok:
                rag_enabled = True
                print("--> RAG update successful, RAG system enabled.")
            else:
                rag_enabled = False
                print("--> RAG update failed, RAG system disabled.")
                return {"status": "failed"}

            return {"status": "success"}
        
        @app.post("/v1/summarypdf")
        async def summary_pdf(request: Request):
            nonlocal summary_enabled, current_rag_summary
            """
            Accepts a JSON body containing 'document_path',
            loads the PDF, and registers it in a RAG index.
            """
            body = await request.json()

            document_path = body.get("document_path")

            # TODO
            # Summary create function
            # rag_update_ok = rag_update_file(document_path)
            rag_update_ok = False

            if rag_update_ok:
                summary_enabled = True
                print("--> RAG update successful, RAG system enabled.")
            else:
                summary_enabled = False
                print("--> RAG update failed, RAG system disabled.")
                return {"status": "failed"}

            return {"status": "success"}

        def generate_chat_completion_chunks(text):
            response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            
            # Split text into chunks
            chunks = text.splitlines(keepends=True)
            
            for i, chunk in enumerate(chunks):
                # Create ChatCompletionChunk object
                chunk_obj = ChatCompletionChunk(
                    id=response_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="generic",
                    choices=[
                        Choice(
                            index=0,
                            delta=ChoiceDelta(content=chunk + " "),
                            finish_reason=None if i < len(chunks) - 1 else "stop"
                        )
                    ]
                )
                yield chunk_obj
                time.sleep(0.1)  # simulate streaming delay
        
        async def event_generator(generator, sources=None):
            for element in generator:
                yield element.model_dump_json()

            # After streaming, append sources if present
            if sources:
                sources_text = "\n\n---\nSources:\n" + "\n".join(sources)
                # Yield as a final chunk in OpenAI streaming format
                sources_chunk = ChatCompletionChunk(
                    id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="generic",
                    choices=[
                        Choice(
                            index=0,
                            delta=ChoiceDelta(content=sources_text),
                            finish_reason="stop"
                        )
                    ]
                )
                yield sources_chunk.model_dump_json()

            yield "[DONE]"

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            nonlocal rag_enabled, rag_provider
            try:
                payload = await request.json()

                # Get last user message
                messages = payload.get("messages", [])
                last_message = messages[-1]  # This is a dict: {"role": "...", "content": "..."}
                last_user_message = last_message.get("content", "")

                stream = payload.get("stream", False)

                # ==== Start command control sequence ====
                
                tokens = last_user_message.split()
                command = tokens[0].lower()
                args = tokens[1:]

                if command == "/help":
                    # send back test message
                    command_list = (
                                    "| Command | Description |\n"
                                    "|---------|-------------|\n"
                                    "| `/chatwithfile <filename.pdf>` | Load a PDF file and chat with it |\n"
                                    "| `/summarize <filename.pdf or URL>` | Summarize a document or website and chat with the summary |\n"
                                    "| `/chatwithwebsite <URL>` | Load a website and chat with it |\n"
                                    "| `/chatwithwebsite /deep <URL>` | Load a website, visit all sublinks, and chat with it |\n"
                                    "| `/forgetcontext` | Disable background injection of content |\n"
                                    "| `/persist` | (Description not provided — likely enable persistent context) |\n"
                                    "| `/enablememory` or `/disablememory` | Enable or disable memory |\n"
                                    )

                    stream_response = generate_chat_completion_chunks(command_list)
                    return EventSourceResponse(event_generator(stream_response))
                
                if command == "/chatwithfile":
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /chatwithfile <Path to PDF or txt file>")
                        return EventSourceResponse(event_generator(stream_response))
                   
                    else:
                        rag_update_ok = rag_update_file(args[0])

                        if rag_update_ok:
                            rag_enabled = True
                            stream_response = generate_chat_completion_chunks(f"Ready, you can now chat with {args[0]}!")
                            return EventSourceResponse(event_generator(stream_response))
                        else:
                            rag_enabled = False
                            stream_response = generate_chat_completion_chunks(f"There was an error while reading the document {args[0]}, please try again.")
                            return EventSourceResponse(event_generator(stream_response))
                        
                if command == "/chatwithwebsite":
                    if "/deep" in last_user_message:
                        # If deep flag -> args must be 2
                        deep_crawl = True

                        if len(args) != 2:
                            stream_response = generate_chat_completion_chunks("Usage: /chatwithwebsite /deep <URL>")
                            return EventSourceResponse(event_generator(stream_response))

                        com_index = 1
    
                    else:
                        deep_crawl = False

                        if len(args) != 1:
                            stream_response = generate_chat_completion_chunks("Usage: /chatwithwebsite <URL>")
                            return EventSourceResponse(event_generator(stream_response))

                        com_index = 0

                    rag_update_ok = rag_update_web(args[com_index], deep_crawl)

                    if rag_update_ok:
                        rag_enabled = True
                        stream_response = generate_chat_completion_chunks(f"Ready, you can now chat with {args[com_index]}!")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
                        rag_enabled = False
                        stream_response = generate_chat_completion_chunks(f"There was an error while reading the document {args[com_index]}, please try again.")
                        return EventSourceResponse(event_generator(stream_response))
                        
                # ========================================

                rag_sources = None

                if rag_enabled:
                    # --- Inject RAG context before forwarding ---
                    search_query = last_user_message

                    # Optimize query if enabled
                    if self.enable_query_opt:
                        print("--> Starting query optimization.")

                        instructions_query_opt =   f"""Task:\n
                            - You are a query optimization assistant.\n
                            - Your goal is to transform a user’s natural-language query into a rewritten query that is optimized for semantic similarity search in a vector database.\n
                            Rewrite Requirements:\n
                            - Preserve the user’s intent.\n
                            - Identify the focus topic of the users input and reduce the query to this topic\n
                            - Make it more specific, detailed, and semantically rich.\n
                            - Add related key concepts, synonyms, and domain-specific terminology.\n
                            - Use concise phrases, not full sentences.\n
                            - Remove conversational filler (e.g., “Can you tell me…”).\n
                            Output Format:\n
                            - Provide only the rewritten query—no explanations or extra text.\n
                            User Query:\n
                            {search_query}\n
                            Optimized Similarity Search Query:\n"""
                        
                        input_msg_query_opt = [
                                {
                                    "role": "user",
                                    "content": instructions_query_opt,
                                }
                            ]
                        
                        response_query_opt = client.chat.completions.create(
                                                model=payload.get("model", "generic"),
                                                messages=input_msg_query_opt,
                                                stream=False,
                                                temperature=0.1,
                                            )

                        search_query = response_query_opt.choices[0].message.content

                        print(f"--> Optimized search query: {search_query}")
                        

                    # Query Vectorstore
                    rag_output = rag_provider.search_knn(search_query, num_chunks=self.rag_max_chunks)

                    rag_context = "The following parts of a document or website should be considered when generating responses and/or answers to the users questions:\n"
                    rag_sources = []

                    time.sleep(0.01)

                    num = 1
                    for result in rag_output:
                        if result.get("similarity", 0) < self.rag_score_thresh:
                            # Skip source if similarity is too low
                            continue

                        rag_context += f"[\n{num}:\n"
                        rag_context += result.get("chunk", "")

                        # Include source meta info for output
                        source_info     = result.get("source_info")
                        source_position = result.get("source_position")

                        if source_info is not None or source_position is not None:
                            if source_position != 0:
                                rag_sources.append(f"{num}: {source_info}, Page: {source_position}")
                            else:
                                rag_sources.append(f"{num}: {source_info}")

                        rag_context += f"\n],\n"
                        num += 1

                    if len(rag_sources) == 0:
                        rag_context += f"There are no information in the document that can answer the user's question. Do not answer anything that you think it  may be correct.\n"
                    else:
                        rag_context += f"All of the parts of a document or website should only be used if it is helpful in answering the user's question. Do not output filenames or URLs that may be included in the context.\n"

                    injected_message = {
                        "role": "user",
                        "content": rag_context
                    }

                    payload["messages"].insert(0, injected_message)  # insert at top
                    # OR: payload["messages"].append(injected_message)

                # Streaming mode
                if stream:
                    stream_response = client.chat.completions.create(**payload)
                    return EventSourceResponse(event_generator(stream_response, rag_sources))

                # Non-streaming mode
                response = client.chat.completions.create(**payload)

                # Append RAG sources
                if rag_enabled:
                    try:
                        response.choices[0].message.content += "\n\n---\nSources:\n"
                        for source in rag_sources:
                            response.choices[0].message.content += f"{source}\n"
                    except Exception as e:
                        print(f"--> Failed to append RAG sources: {e}")

                    return JSONResponse(response.model_dump_json())

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        uvicorn.run(app, host="0.0.0.0", port=int(self.rag_proxy_serve_port))

    def get_rag_proxy_serve_port(self):
        return self.rag_proxy_serve_port
    
    def start(self):
        # Starts the server in a non-blocking separate process.
        if self.process is None or not self.process.is_alive():
            self.process = Process(target=self._run_server, daemon=True)
            self.process.start()
            print(f"--> RAG server started in separate process (PID={self.process.pid})")
        else:
            print("--> RAG server is already running.")

    def stop(self):
        # Stops the server process if running.
        if self.process and self.process.is_alive():
            print(f"--> Stopping RAG server (PID={self.process.pid})...")
            self.process.terminate()
            self.process.join()
            print("--> RAG server stopped.")
        else:
            print("-->  No running RAG server to stop.")
