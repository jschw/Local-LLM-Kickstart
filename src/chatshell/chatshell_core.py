from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
import asyncio, uvicorn
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pathlib import Path
from urllib.parse import urlparse
from PyPDF2 import PdfReader
import json
import os, appdirs, time, json
import uuid
import re
from multiprocessing import Process, Event
import pyperclip


class Chatshell:

    def __init__(self, termux_paths=False):
        self.version = "0.1.0"
        self.process = None
        self.shutdown_event = None

        self.termux = termux_paths

        CONFIG_DIR                  = Path(appdirs.user_config_dir(appname='chatshell'))
        self.chatshell_config_path  = CONFIG_DIR / 'chatshell_server_config.json'
        self.chatshell_config       = None
        self.doc_base_dir           = None
        self.website_crawl_depth    = 1
        self.rag_chunk_count        = 4
        self.chatshell_proxy_serve_port   = 0
        self.llm_server_port        = 0

        self.rag_score_thresh       = 0.5
        self.rag_max_chunks         = 10

        self.load_config()

    def load_config(self):
        """
        Load and parse the chatshell_config.json file into structured variables.
        """
        try:
            if not self.chatshell_config_path.exists():
                # Create llm config file if not existing
                # Template content of the llm_server_config.json
                if self.termux:
                    doc_base_dir_tmp = "~/storage/shared/chatshell/Documents"
                else:
                    doc_base_dir_tmp = "~/chatshell/Documents"

                tmp_chatshell_config = {
                    "rag-document-base-dir": doc_base_dir_tmp,
                    "website-crawl-depth": "2",
                    "rag-chunk-count": "5",
                    "chatshell-proxy-server-port": "4001",
                    "inference-endpoint-base-url": "http://localhost:4000/v1",
                    "use-openai-public-api": "False",
                    "openai-api-token": "mytoken"
                    }

                with self.chatshell_config_path.open('w') as f:
                    json.dump(tmp_chatshell_config, f, indent=4)

            with open(self.chatshell_config_path, "r") as f:
                self.chatshell_config = json.load(f)
                self.chatshell_proxy_serve_port   = self.chatshell_config["chatshell-proxy-server-port"]
                self.endpoint_base_url      = self.chatshell_config["inference-endpoint-base-url"]
                self.doc_base_dir           = Path(os.path.expanduser(self.chatshell_config["rag-document-base-dir"]))
                self.website_crawl_depth    = int(self.chatshell_config["website-crawl-depth"])
                self.rag_chunk_count        = int(self.chatshell_config["rag-chunk-count"])
                self.use_openai_api         = json.loads(str(self.chatshell_config["use-openai-public-api"]).lower())
                self.openai_api_token        = self.chatshell_config["openai-api-token"]

        except Exception as e:
            print(f"Failed to load config file {self.chatshell_config_path}: {e}")
            self.llm_server_config = None
    
    def _run_server(self, shutdown_event):
        self.doc_base_dir.mkdir(parents=True, exist_ok=True)

        self.command_list = [
            "/chatwithfile",
            "/chatwithwebsite",
            "/forgetcontext"
        ]

        # Configure OpenAI API key
        if self.use_openai_api:
            client = OpenAI(
                api_key=self.openai_api_token
            )
        else:
            client = OpenAI(
                api_key="dummy",  # not used locally
                base_url=self.endpoint_base_url  # llama.cpp server endpoint
            )

        app = FastAPI(
            title="Open Prompt Proxy",
            description="A drop-in compatible OpenAI API wrapper that logs prompts and forwards requests.",
            version=self.version
        )

        from .vectorstore import ChatshellVectorsearch
        rag_provider    = ChatshellVectorsearch()
        rag_enabled     = False
        context_enabled = False

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

        def is_url(path_or_url):
            """
            Returns True if the input is an HTTP/HTTPS URL, False if it's a file path.
            """
            try:
                result = urlparse(path_or_url)
                return result.scheme in ("http", "https")
            except Exception:
                return False
        
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

        def get_text_clipboard():
            try:
                content = pyperclip.paste()
            except Exception:
                print("Clip error")
                return None

            if content.strip() == "":
                print("Clip empty")
                return None
            else:
                return content

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            nonlocal rag_enabled, rag_provider, context_enabled
            try:
                payload = await request.json()

                # Get last user message
                messages = payload.get("messages", [])

                # Remove any message whose content matches a command in command list, and the following message
                # EXCEPT if the command is in the last message.
                i = 0
                while i < len(messages) - 1:  # never remove the last message
                    content = messages[i].get("content", "")
                    if any(content.strip().startswith(cmd) for cmd in self.command_list):
                        del messages[i]
                        # After deletion, the next message is now at index i (unless it was the last)
                        if i < len(messages) - 1:
                            del messages[i]
                        # Do not increment i, as the next message is now at the same index
                    else:
                        i += 1

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
                                    "| `/summarize /clipboard` | Summarize the contents of the clipboard and chat with the summary |\n"
                                    "| `/chatwithwebsite <URL>` | Load a website and chat with it |\n"
                                    "| `/chatwithwebsite /deep <URL>` | Load a website, visit all sublinks, and chat with it |\n"
                                    "| `/chatwithclipbrd` | Fetch content from clipboard and chat with the contents |\n"
                                    "| `/addclipboard` | Add the content of the clipboard to every message in the chat |\n"
                                    "| `/forgetcontext` | Disable background injection of every kind of content |\n"
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
                    
                if command == "/chatwithclipbrd":

                    # Handle as Clipboard content
                    clip_content = get_text_clipboard()
                    if clip_content != None:
                        # RAG update with clipboard content
                        rag_update_ok = rag_provider.init_vectorstore_str(clip_content)
                    else:
                        stream_response = generate_chat_completion_chunks(f"The clipboard is empty or not valid text content.")
                        return EventSourceResponse(event_generator(stream_response))
                    
                    if rag_update_ok:
                        rag_enabled = True
                        stream_response = generate_chat_completion_chunks("Ready, you can now chat with the clipboard content!")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
                        rag_enabled = False
                        stream_response = generate_chat_completion_chunks("There was an error while clipboard content, please try again.")
                        return EventSourceResponse(event_generator(stream_response))
                    
                if command == "/summarize":
                    additional_prompt = ""
                    use_add_prompt = False

                    # Regex to match /prompt:"...prompt text..." at the end
                    prompt_pattern = r'/prompt:"([^"]+)"\s*$'

                    # Check if /prompt is present at the end of the message
                    prompt_match = re.search(prompt_pattern, last_user_message)
                    if prompt_match:
                        additional_prompt = prompt_match.group(1)
                        use_add_prompt = True
                        # Remove the /prompt:"..." part from args for further processing
                        # Remove last arg if it is /prompt:"..."
                        if args and args[-1].startswith('/prompt:'):
                            args = args[:-1]

                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /summarize <Path to PDF URL> (/prompt:\"Additional instructions for summarization\")")
                        return EventSourceResponse(event_generator(stream_response))
                   
                    else:
                        chunk_list = []
                        input_path_url = args[0]

                        # Check if argument is clipboard content
                        if "/clipboard" in input_path_url:
                            # Handle as Clipboard content
                            clip_content = get_text_clipboard()
                            if clip_content != None:
                                chunk_list = [clip_content]
                            else:
                                stream_response = generate_chat_completion_chunks(f"The clipboard is empty or not valid text content.")
                                return EventSourceResponse(event_generator(stream_response))

                        # Use is_url to check if input_path_url is a URL or a file path
                        elif is_url(input_path_url):
                            # Handle as URL
                            # Crawl website
                            print(f"-> Crawling {input_path_url}.")
                            from vectorstore import crawl_website
                            page_contents = crawl_website(input_path_url, 5, max_depth=1)

                            if page_contents is not None and len(page_contents) > 0:

                                for page_text, page_url in page_contents:
                                    chunk_list.append(page_text)

                        else:
                            # Handle as file path
                            doc_current = input_path_url

                            if not os.path.isfile(doc_current):
                                # Document is not available at absolute path, checking rel. path
                                doc_current = os.path.join(self.doc_base_dir, doc_current)
                                if not os.path.isfile(doc_current):
                                    # Document is not available -> return error
                                    print(f"--> Document {input_path_url} not found.")
                                    stream_response = generate_chat_completion_chunks(f"The document {input_path_url} was not found.\nPlease enter a valid document path.")
                                    return EventSourceResponse(event_generator(stream_response))

                            # --> Read PDF pages into chunk list
                            reader = PdfReader(doc_current)

                            # Load each page's text into a list, one entry per page
                            for page in reader.pages:
                                text = page.extract_text()
                                chunk_list.append(text)

                        try:
                            # Create summary
                            print("--> Start summarization...")
                            text_summary = rag_provider.generate_text_summary(chunk_list)
                            print("--> Generated summary chunks.")

                            # Build context
                            instructions_summarization =   f"""Task:\n
                                - You are a summarization assistant.\n
                                - Your goal is to write a summary of a list of given texts that represent a docuemnt.\n
                                Rewrite Requirements:\n
                                - Preserve the information that are given inside the texts\n
                                - Use a neutral language that is well understandable\n
                                - Format your summary well for goo readability\n
                                - Do not refer to this given task\n
                                - Write your summary as a list of points if neccessary\n
                                - Use line breaks if neccessary for longer summaries\n
                                - Use markdown formatting for good readability\n
                                Output Format:\n
                                - Provide only the summary - no explanations or extra text.\n"""

                            if use_add_prompt and additional_prompt:
                                instructions_summarization += f"\nAdditional Prompt:\n{additional_prompt}\n"

                            instructions_summarization += f"Text list:\n{text_summary}\nSummary:\n"

                            # Invoke inference for summarization
                            input_msg_summarization = [
                                    {
                                        "role": "user",
                                        "content": instructions_summarization,
                                    }
                                ]
                        
                            response_summarization = client.chat.completions.create(
                                                    model=payload.get("model", "generic"),
                                                    messages=input_msg_summarization,
                                                    stream=True,
                                                    temperature=0.1,
                                                )
                        except Exception as e:
                            stream_response = generate_chat_completion_chunks(f"There was an error while creating the summary: {str(e)}")
                            return EventSourceResponse(event_generator(stream_response))

                        return EventSourceResponse(event_generator(response_summarization))
                        
                if command == "/addclipboard":
                    # Add all clipboard content to context list
                    context_enabled = True

                    clip_content = get_text_clipboard()
                    if clip_content != None:
                        rag_provider.add_context(clip_content)
                        stream_response = generate_chat_completion_chunks(f"The clipboard content was inserted into context.")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
                        stream_response = generate_chat_completion_chunks(f"The clipboard is empty or not valid text content.")
                        return EventSourceResponse(event_generator(stream_response))
                
                if command == "/forgetcontext":
                    rag_enabled     = False
                    context_enabled = False
                    rag_provider.reset_context()
                    stream_response = generate_chat_completion_chunks("Document or website context is no longer included in chat.")
                    return EventSourceResponse(event_generator(stream_response))
                
                # ========================================

                rag_sources = None

                if rag_enabled:
                    # --- Inject RAG context before forwarding ---
                    search_query = last_user_message

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

                    payload["messages"][-1]["content"] += "\n" + rag_context # insert at end of last user message

                if context_enabled:
                    # Adding context if there is something
                    current_context = rag_provider.get_context()

                    if current_context != "":
                        current_context += f"There is some additional information in the context that can help answer the user's question. Do not refer directly to this context.\n"

                    payload["messages"][-1]["content"] += "\n" + current_context # insert at end of last user message
                
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

        # Starting up uvicorn.Server
        config = uvicorn.Config(app, host="0.0.0.0", port=int(self.chatshell_proxy_serve_port), loop="asyncio")
        server = uvicorn.Server(config)

        async def serve_until_event():
            server_task = asyncio.create_task(server.serve())
            while not shutdown_event.is_set():
                await asyncio.sleep(0.5)
            if server.started:
                # Shutdown if loop was completed
                await server.shutdown()
                return
            await server_task

        try:
            asyncio.run(serve_until_event())
        except Exception as e:
            print(f"Exception in server loop: {e}")

    def get_chatshell_proxy_serve_port(self):
        return self.chatshell_proxy_serve_port
    
    def start(self):
        # Starts the server in a non-blocking separate process.
        if self.process is None or not self.process.is_alive():
            self.shutdown_event = Event()
            self.process = Process(target=self._run_server, args=(self.shutdown_event,))
            self.process.start()
            print(f"--> RAG server started in separate process (PID={self.process.pid})")
        else:
            print("--> RAG server is already running.")

    def stop(self):
        # Stops the server process if running.
        if self.process and self.process.is_alive():
            print(f"--> Stopping RAG server (PID={self.process.pid}), sending shutdown signal...")
            if self.shutdown_event:
                self.shutdown_event.set()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    # Server process hanging -> terminate signal
                    self.process.terminate()
                else:
                    print("--> RAG server stopped gracefully.")
