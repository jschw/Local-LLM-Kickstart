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
import os, appdirs, time, json, uuid, re
from datetime import datetime
from multiprocessing import Process, Event
import pyperclip
from .llm_server import LocalLLMServer
from .context_manager import ContextManager


class Chatshell:

    def __init__(self, termux_paths=False):
        self.version = "0.2.0"
        self.process = None
        self.shutdown_event = None

        self.termux                 = termux_paths

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

        # Start LLM server
        llm_server                  = LocalLLMServer(termux_paths=self.termux)
        llm_config_path        = llm_server.get_llm_config_path()
        llm_server_config_path = llm_server.get_llm_server_config_path()

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

        context_manager = ContextManager()
        context_manager.set_doc_base_dir(self.doc_base_dir)
        # Status variables
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

        def endpoint_avail()->bool:
            if self.use_openai_api == False and llm_server.process_started() == False:
                # OpenAI endpoint turned off and no local endpoint available
                return False
            else:
                return True
            
        def format_model_list(avail_models):
            header = (
                "Available LLM models:\n"
                "| Model name | Port | Path | HF Repo | HF Repo File |\n"
                "|------------|------|------|---------|--------------|\n"
            )
            rows = []
            for model in avail_models:
                name = model.get("name", "")
                port = model.get("port", "")
                path = model.get("model", "")
                hf_repo = model.get("hf-repo", "")
                hf_file = model.get("hf-file", "")
                row = f"| {name} | {port} | {path} | {hf_repo} | {hf_file} |"
                rows.append(row)
            return header + "\n".join(rows)

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            nonlocal rag_enabled, context_enabled, llm_server

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
                first_message = messages[0].get("content", "")
                last_user_message = last_message.get("content", "")

                stream = payload.get("stream", False)

                # Set shellmode state for this request
                shellmode_active = False
                if first_message == "/shellmode":
                    shellmode_active = True

                # ==== Start command control sequence ====

                tokens = last_user_message.split()
                command = tokens[0].lower()
                args = tokens[1:]

                if command == "/help":
                    # send back test message
                    command_list = (
                                    "| Command | Description |\n"
                                    "|---------|-------------|\n"
                                    "| `/help` | Show this help message |\n"
                                    "| `/chatwithfile <filename.pdf>` | Load a PDF or text file and chat with it |\n"
                                    "| `/chatwithwebsite <URL>` | Load a website and chat with it |\n"
                                    "| `/chatwithwebsite /deep <URL>` | Load a website, visit all sublinks, and chat with it |\n"
                                    "| `/chatwithclipbrd` | Fetch content from clipboard and chat with the contents |\n"
                                    "| `/summarize <filename.pdf or URL>` | Summarize a document or website and chat with the summary |\n"
                                    "| `/summarize /clipboard` | Summarize the contents of the clipboard and chat with the summary |\n"
                                    "| `/addclipboard` | Add the content of the clipboard to every message in the chat |\n"
                                    "| `/forgetcontext` | Disable background injection of every kind of content |\n"
                                    "| `/forgetall` | Disable RAG and all inserted contexts |\n"
                                    "| `/forgetctx` | Disable inserted context only |\n"
                                    "| `/forgetdoc` | Disable RAG (document/website context) only |\n"
                                    "| `/updatemodels` | Update the LLM model catalog from GitHub |\n"
                                    "| `/startendpoint <Endpoint config name>` | Start a specific LLM endpoint |\n"
                                    "| `/restartendpoint <Endpoint config name>` | Restart a specific LLM endpoint |\n"
                                    "| `/stopendpoint <Endpoint config name>` | Stop a specific LLM endpoint |\n"
                                    "| `/stopallendpnts` | Stop all LLM inference endpoints |\n"
                                    "| `/llmstatus` | Show the status of local LLM inference endpoints |\n"
                                    "| `/setautostartendpoint <LLM endpoint name>` | Set a specific LLM endpoint for autostart |\n"
                                    "| `/listendpoints` | List all available LLM endpoint configs |\n"
                                    "| `/shellmode` | Activate shell mode for this chat (no LLM interaction) |\n"
                                    "| `/exit` | Quit chatshell server |\n"
                                    )

                    stream_response = generate_chat_completion_chunks(command_list)
                    return EventSourceResponse(event_generator(stream_response))

                if command == "/chatwithfile":
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /chatwithfile <Path to PDF or txt file>")
                        return EventSourceResponse(event_generator(stream_response))

                    else:
                        rag_update_ok, output_msg = context_manager.rag_update_file(args[0])
                        rag_enabled = rag_update_ok

                        stream_response = generate_chat_completion_chunks(output_msg)
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/chatwithwebsite":
                    if "/deep" in last_user_message:
                        # If deep flag -> args must be 2
                        deep_crawl = True

                        if len(args) != 2:
                            stream_response = generate_chat_completion_chunks("Usage: /chatwithwebsite /deep <URLs>")
                            return EventSourceResponse(event_generator(stream_response))

                        com_index = 1

                    else:
                        deep_crawl = False

                        if len(args) != 1:
                            stream_response = generate_chat_completion_chunks("Usage: /chatwithwebsite <URLs>")
                            return EventSourceResponse(event_generator(stream_response))

                        com_index = 0

                    rag_update_ok = context_manager.rag_update_web(args[com_index], deep_crawl)
                    rag_enabled = rag_update_ok

                    if rag_update_ok:
                        stream_response = generate_chat_completion_chunks(f"Ready, you can now chat with {args[com_index]}!")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
                        stream_response = generate_chat_completion_chunks(f"There was an error while reading the document {args[com_index]}, please try again.")
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/chatwithclipbrd":

                    # Handle as Clipboard content
                    clip_content = get_text_clipboard()
                    if clip_content != None:
                        # RAG update with clipboard content
                        rag_update_ok = context_manager.rag_update_clipbrd(clip_content)
                        rag_enabled = rag_update_ok 
                    else:
                        stream_response = generate_chat_completion_chunks(f"The clipboard is empty or not valid text content.")
                        return EventSourceResponse(event_generator(stream_response))

                    if rag_update_ok:
                        stream_response = generate_chat_completion_chunks("Ready, you can now chat with the clipboard content!")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
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
                        if not endpoint_avail():
                            # No public OpenAI connection configured and local endpoint not available
                            stream_response = generate_chat_completion_chunks("There is no LLM inference endpoint available. Please configure first and try again.")
                            return EventSourceResponse(event_generator(stream_response))
                
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
                            text_summary = context_manager.rag_provider.generate_text_summary(chunk_list)
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
                        context_manager.add_context(clip_content)
                        stream_response = generate_chat_completion_chunks(f"The clipboard content was inserted into context.")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
                        stream_response = generate_chat_completion_chunks(f"The clipboard is empty or not valid text content.")
                        return EventSourceResponse(event_generator(stream_response))
                
                if command == "/forgetall":
                    # Disable RAG and other inserted contexts
                    rag_enabled     = False
                    context_enabled = False
                    context_manager.reset_context()
                    stream_response = generate_chat_completion_chunks("Document or website context is no longer included in chat.")
                    return EventSourceResponse(event_generator(stream_response))
                
                if command == "/forgetctx":
                    # Disable other inserted contexts
                    context_enabled = False
                    context_manager.reset_context()
                    stream_response = generate_chat_completion_chunks("Context is no longer included in chat.")
                    return EventSourceResponse(event_generator(stream_response))
                
                if command == "/forgetdoc":
                    # Disable RAG
                    rag_enabled     = False
                    context_manager.reset_rag_context()
                    stream_response = generate_chat_completion_chunks("Document or website context is no longer included in chat.")
                    return EventSourceResponse(event_generator(stream_response))
                
                if command == "/updatemodels":
                    # Fetch current version of model catalog from github
                    update_models_ok = llm_server.update_model_catalog()

                    if update_models_ok:
                        # Fetch model list and output
                        models_avail = llm_server.get_endpoints()
                        stream_response = generate_chat_completion_chunks(format_model_list(models_avail))
                        return EventSourceResponse(event_generator(stream_response))

                    else:
                        stream_response = generate_chat_completion_chunks(f"Updating the LLM model catalog failed.")
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/startendpoint":
                    # Starts a specific LLM endpoint
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /startendpoint <Endpoint config name>")
                        return EventSourceResponse(event_generator(stream_response))
                   
                    else:
                        start_endpoint_ok, output = llm_server.create_endpoint(args[0])

                        stream_response = generate_chat_completion_chunks(output)
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/restartendpoint":
                    # Restart a certain LLM inference endpoint
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /restartendpoint <Endpoint config name>")
                        return EventSourceResponse(event_generator(stream_response))
                   
                    else:
                        start_endpoint_ok, output = llm_server.restart_process(args[0])

                        stream_response = generate_chat_completion_chunks(output)
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/stopendpoint":
                    # Stop a certain LLM inference endpoint
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /stopendpoint <Endpoint config name>")
                        return EventSourceResponse(event_generator(stream_response))
                   
                    else:
                        stop_endpoint_ok, output = llm_server.stop_process(args[0])

                        stream_response = generate_chat_completion_chunks(output)
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/stopallendpnts":
                    # Stop all LLM inference endpoints
                    output = llm_server.stop_all_processes()

                    stream_response = generate_chat_completion_chunks("\n".join(output))
                    return EventSourceResponse(event_generator(stream_response))
                
                if command == "/llmstatus":
                    # Show the current status of local LLM inference endpoints
                    endpoint_processes = llm_server.list_processes()
                    print(endpoint_processes)

                    if len(endpoint_processes) > 0:
                        header = (
                            "| Inference Endpoints |\n"
                            "|--------------|\n"
                        )

                        rows = []
                        for endpoint in endpoint_processes:
                            row = f"| {endpoint} |"
                            rows.append(row)

                        header = header + "\n".join(rows)

                        print(header)

                        stream_response = generate_chat_completion_chunks(header)
                    else:
                        stream_response = generate_chat_completion_chunks("There are currently no running LLM inference endpoints.")

                    return EventSourceResponse(event_generator(stream_response))
                
                if command == "/setautostartendpoint":
                    # Set a specific LLM endpoint for autostart at application startup
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /setautostartendpoint <LLM endpoint name>")
                        return EventSourceResponse(event_generator(stream_response))
                   
                    else:
                        set_as_endpoint_ok = llm_server.set_autostart_endpoint(args[0])

                        if set_as_endpoint_ok:
                            stream_response = generate_chat_completion_chunks(f"The LLM endpoint '{args[0]}' was set correcty and will be started automatically on next start of chatshell.")
                            return EventSourceResponse(event_generator(stream_response))
                        else:
                            stream_response = generate_chat_completion_chunks(f"There was an error setting the LLM endpoint '{args[0]}' for automatic startup.\nEnsure that the model file exists at the path in configuration.")
                            return EventSourceResponse(event_generator(stream_response))

                if command == "/listendpoints":
                    # Outputs all available LLM endpoint configs
                    models_avail = llm_server.get_endpoints()
                    stream_response = generate_chat_completion_chunks(format_model_list(models_avail))
                    return EventSourceResponse(event_generator(stream_response))
                
                if command == "/status":
                    # Outputs the status of the system overall context + RAG + LLM server
                    status_message = (
                        "## Chatshell system status\n"
                        "### LLM Endpoint Server\n")
                    
                    # Add information about running inference endpoints
                    endpoint_processes = llm_server.list_processes()

                    if len(endpoint_processes) > 0:
                        tmp_endpoints = (
                            "| Inference Endpoints |\n"
                            "|--------------|\n"
                        )

                        rows = []
                        for endpoint in endpoint_processes:
                            row = f"| {endpoint} |"
                            rows.append(row)

                        tmp_endpoints = tmp_endpoints + "\n".join(rows)

                        status_message += tmp_endpoints
                        status_message += "\n"

                    else:
                        status_message += "- No running inference endpoints\n"

                    # Add information about RAG system status
                    status_message += "### RAG System\n"
                    tmp_rag = []
                    # RAG Status
                    tmp_rag = []
                    tmp_rag.append(f"- **Enabled:** {'Yes' if rag_enabled else 'No'}")
                    rag_mode_str = {0: "File", 1: "Web", 2: "Clipboard"}.get(context_manager.rag_mode, "Unknown")
                    if rag_enabled:
                        tmp_rag.append(f"- **Mode:** {rag_mode_str}")
                    tmp_rag.append(f"- **Loaded Content:** {context_manager.rag_content_list if context_manager.rag_content_list else 'None'}")
                    tmp_rag.append(f"- **Last Update:** {context_manager.rag_update_time if context_manager.rag_update_time else 'Never'}")
                    status_message += "\n".join(tmp_rag) + "\n"

                    # Add information about context status
                    status_message += "### Additional context\n"
                    tmp_ctx = []
                    tmp_ctx.append(f"- **Enabled:** {'Yes' if context_enabled else 'No'}")
                    tmp_ctx.append(f"- **Last Update:** {context_manager.context_update_time if context_manager.context_update_time else 'Never'}")
                    status_message += "\n".join(tmp_ctx) + "\n"

                    stream_response = generate_chat_completion_chunks(status_message)
                    return EventSourceResponse(event_generator(stream_response))
 

                if command == "/shellmode":
                    # Activate shell mode for specific chat by inserting the keyword
                    stream_response = generate_chat_completion_chunks(f"This chat is now marked as shell-chat, no LLM interaction will be performed on future inputs.")
                    return EventSourceResponse(event_generator(stream_response))

                if command == "/exit":
                    # Quit chatshell server
                    quit()
                
                # ========================================

                if shellmode_active:
                    stream_response = generate_chat_completion_chunks("Shell mode is enabled for this chat. You can use this chat for communication with chatshell itself - your messages are not redirected to a LLM inference endpoint.\nIf you want to communicate with an LLM, please open a new chat conversion.")
                    return EventSourceResponse(event_generator(stream_response))
                
                if not endpoint_avail():
                    # No public OpenAI connection configured and local endpoint not available
                    stream_response = generate_chat_completion_chunks("There is no LLM inference endpoint available. Please configure first and try again.")
                    return EventSourceResponse(event_generator(stream_response))

                rag_sources = None

                if rag_enabled:
                    # --- Inject RAG context before forwarding ---
                    search_query = last_user_message

                    # Query Vectorstore
                    rag_output = context_manager.rag_provider.search_knn(search_query, num_chunks=self.rag_max_chunks)

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
                    current_context = context_manager.get()

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

