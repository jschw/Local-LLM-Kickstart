from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
import asyncio, uvicorn
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pathlib import Path
from urllib.parse import urlparse
import os, appdirs, time, json, uuid
from multiprocessing import Process, Event
import pyperclip
from difflib import SequenceMatcher
from .llm_server import LocalLLMServer
from .context_manager import ContextManager


class Chatshell:

    def __init__(self, termux_paths=False):
        self.version = "0.4.0"
        self.process = None
        self.shutdown_event = None

        self.termux                 = termux_paths

        CONFIG_DIR                  = Path(appdirs.user_config_dir(appname='chatshell'))
        self.chatshell_config_path  = CONFIG_DIR / 'chatshell_server_config.json'
        self.chatshell_config       = None
        self.doc_base_dir           = None
        self.output_base_dir        = None
        self.rag_document_user_dirs = []
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
                    self.output_base_dir = Path(os.path.expanduser("~/storage/shared/chatshell/Output"))
                else:
                    doc_base_dir_tmp = "~/chatshell/Documents"
                    self.output_base_dir = Path(os.path.expanduser("~/chatshell/Output"))
 
                tmp_chatshell_config = {
                    "rag-document-base-dir": doc_base_dir_tmp,
                    "rag-document-user-dirs": [],
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

                # Parse rag-document-user-dirs as a list if present, else empty list
                user_dirs = self.chatshell_config.get("rag-document-user-dirs", None)
                if user_dirs is None or user_dirs == "":
                    self.rag_document_user_dirs = []
                elif isinstance(user_dirs, list):
                    self.rag_document_user_dirs = [Path(os.path.expanduser(d)) for d in user_dirs]
                else:
                    raise ValueError("rag-document-user-dirs must be a JSON list of directory paths.")

                self.website_crawl_depth    = int(self.chatshell_config["website-crawl-depth"])
                self.rag_chunk_count        = int(self.chatshell_config["rag-chunk-count"])
                self.use_openai_api         = json.loads(str(self.chatshell_config["use-openai-public-api"]).lower())
                self.openai_api_token        = self.chatshell_config["openai-api-token"]

        except Exception as e:
            print(f"Failed to load config file {self.chatshell_config_path}: {e}")
            self.llm_server_config = None

    def _run_server(self, shutdown_event):
        self.doc_base_dir.mkdir(parents=True, exist_ok=True)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.command_list = [
            "/help",
            "/filechat",
            "/webchat",
            "/clipchat",
            "/addclipboard",
            "/savetask",
            "/listtasks",
            "/taskinfo",
            "/deletetask",
            "/forgetall",
            "/forgetctx",
            "/forgetdoc",
            "/updatemodels",
            "/startendpoint",
            "/restartendpoint",
            "/stopendpoint",
            "/stopallendpnts",
            "/llmstatus",
            "/setautostartendpoint",
            "/listendpoints",
            "/status",
            "/shellmode",
            "/version",
            "/exit",
            "/addinstruction",
            "/saveinstruction",
            "/loadinstruction",
            "/save"
        ]

        # Start LLM server
        llm_server             = LocalLLMServer(termux_paths=self.termux)
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
        
        def find_similar_command(input_command: str, threshold: float = 0.82):
            if not input_command or not input_command.startswith("/"):
                return None, 0.0

            best_command = None
            best_score = 0.0

            for registered_command in self.command_list:
                score = SequenceMatcher(None, input_command, registered_command).ratio()
                if score > best_score:
                    best_score = score
                    best_command = registered_command

            if best_score >= threshold:
                return best_command, best_score

            return None, best_score

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
        
        def generate_text_summary_chunks(sources:list)->str:
            list_input_sources = []
            list_input_content = []

            for source in sources:
                # process all inputs for summarization
                summary_result , summary_generation_ok = context_manager.summarize_files_urls(source)

                if summary_generation_ok:
                    list_input_sources.append(source)
                    list_input_content.append(summary_result)
            
            if len(list_input_content) == 0:
                return ""
            
            try:
                # Build context for generating a text from summary chunks
                instructions_summarization =   f"""Task:\n
                    - You are a summarization assistant.\n
                    - Your goal is to write a summary of a list of given texts that represent a docuemnt.\n
                    Summarization Requirements:\n
                    - Preserve the information that are given inside the texts\n
                    - Use a neutral language that is well understandable\n
                    - Format your summary well for goo readability\n
                    - Do not refer to this given task\n
                    - Write your summary as a list of points if neccessary\n
                    - Use line breaks if neccessary for longer summaries\n
                    - Use markdown formatting for good readability\n
                    Output Format:\n
                    - Provide only the summary - no explanations or extra text.\n"""

                if context_manager.summarize_additional_prompt != "":
                    instructions_summarization += f"\nAdditional Prompt:\n{context_manager.summarize_additional_prompt}\n"

                instructions_summarization += "\nText list:\n"

                # Assemble summary content
                for i in range(len(list_input_content)):
                    instructions_summarization += f"\nText source: {list_input_sources[i]}\n"
                    instructions_summarization += f"Content to summarize: {list_input_content[i]}\n"
                    instructions_summarization += "###\n"

                instructions_summarization += "\nSummary:\n"

                # Invoke inference for summarization
                input_msg_summarization = [
                        {
                            "role": "user",
                            "content": instructions_summarization,
                        }
                    ]
                return input_msg_summarization
                
            except Exception as e:
                return ""

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

                # Store current payload to context manager
                context_manager.payload = payload

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

                if tokens:
                    # First part of message before first whitespace
                    command = tokens[0].lower()
                    # All following parts after first whitespace
                    args = tokens[1:]
                else:
                    command = ""
                    args = []

                if command and command.startswith("/") and command not in self.command_list:
                    similar_command, similarity = find_similar_command(command)
                    if similar_command:
                        similarity_message = (
                            f"The input '{command}' is not a valid command. It is similar to '{similar_command}'. "
                            "Please check the command spelling and correct it if you want to execute a command."
                        )
                        stream_response = generate_chat_completion_chunks(similarity_message)
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/help":
                    # send back test message
                    command_list = (
                                    "| Command | Description |\n"
                                    "|---------|-------------|\n"
                                    "| `/help` | Show this help message |\n"
                                    "| `/filechat <filename.pdf>` | Load a PDF or text file and chat with it |\n"
                                    "| `/webchat <URL>` | Load a website and chat with it |\n"
                                    "| `/webchat /deep <URL>` | Load a website, visit all sublinks, and chat with it |\n"
                                    "| `/clipchat` | Fetch content from clipboard and chat with the contents |\n"
                                    "| `/summarize <filename.pdf or URL>` | Summarize a document or website and chat with the summary |\n"
                                    "| `/summarize /clipboard` | Summarize the contents of the clipboard and chat with the summary |\n"
                                    "| `/summarize /setprompt \"Additional prompt for summary\"` | Add an additional prompt for customizing your summary |\n"
                                    "| `/addclipboard` | Add the content of the clipboard to every message in the chat |\n"
                                    "| `/savetask /<Task type> <Task name>` | Save the current task (file, web, summarize) |\n"
                                    "| `/runtask <Task name>` | Load and run a saved task |\n"
                                    "| `/reply <msg_num> "<instruction>"` | Apply an instruction to a previous message and run completion |\n"
                                    "| `/listtasks` | List all saved tasks |\n"
                                    "| `/taskinfo <Task name>` | Show detailed info for a specific task |\n"
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

                if command == "/filechat":
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /filechat <Path to PDF or txt file>")
                        return EventSourceResponse(event_generator(stream_response))

                    else:
                        rag_update_ok, output_msg = context_manager.rag_update_file(args[0])
                        rag_enabled = rag_update_ok

                        stream_response = generate_chat_completion_chunks(output_msg)
                        return EventSourceResponse(event_generator(stream_response))
                    
                if command == "/dirchat":
                    if len(args) < 1 or len(args) > 2:
                        stream_response = generate_chat_completion_chunks("Usage: /dirchat <Path to directory with documents> [filetype]")
                        return EventSourceResponse(event_generator(stream_response))

                    else:
                        dir_path = args[0]
                        filetype = args[1] if len(args) == 2 else None
                        rag_update_ok, output_msg = context_manager.rag_update_directory(dir_path, filetype)
                        rag_enabled = rag_update_ok

                        if rag_update_ok:
                            stream_response = generate_chat_completion_chunks(f"Ready, you can now chat with the files in {dir_path}!\n{output_msg}")
                            return EventSourceResponse(event_generator(stream_response))
                        else:
                            stream_response = generate_chat_completion_chunks(f"There was an error while creating the vectorstore index, please try again.")
                            return EventSourceResponse(event_generator(stream_response))

                if command == "/msg2rag":
                    # Create the vectorstore index to chat with from saved message in the output folder
                    dir_path = str(self.output_base_dir)
                    filetype = "json"
                    rag_update_ok, output_msg = context_manager.rag_update_directory(dir_path, filetype)
                    rag_enabled = rag_update_ok

                    if rag_update_ok:
                        stream_response = generate_chat_completion_chunks(f"Ready, you can now chat with your saved conversations!")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
                        stream_response = generate_chat_completion_chunks(f"There was an error while creating the vectorstore index from your saved conversations.")
                        return EventSourceResponse(event_generator(stream_response))
                

                if command == "/webchat":
                    if args[0] == "/deep":
                        # If deep flag -> args must be 2
                        deep_crawl = True

                        if len(args) != 2:
                            stream_response = generate_chat_completion_chunks("Usage: /webchat /deep <URLs>")
                            return EventSourceResponse(event_generator(stream_response))

                        com_index = 1

                    else:
                        deep_crawl = False

                        if len(args) != 1:
                            stream_response = generate_chat_completion_chunks("Usage: /webchat <URLs>")
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

                if command == "/clipchat":
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

                    if args[0] == "/setprompt":
                        # Add additional prompt for summarization
                        context_manager.summarize_additional_prompt = " ".join(last_user_message.split()[2:]).strip().replace("\"", "")
                        stream_response = generate_chat_completion_chunks("Additional prompt for summarization set, you can now request a summary.")
                        return EventSourceResponse(event_generator(stream_response))

                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /summarize <Path to PDF URL>\nYou can also set an additional prompt with /summarize /setprompt \"Additional instructions for summarization\" before invoking a summary.")
                        return EventSourceResponse(event_generator(stream_response))
                
                    else:
                        if not endpoint_avail():
                            # No public OpenAI connection configured and local endpoint not available
                            stream_response = generate_chat_completion_chunks("There is no LLM inference endpoint available. Please configure first and try again.")
                            return EventSourceResponse(event_generator(stream_response))

                        context_manager.summarize_input = args[0]
                        input_path_urls = context_manager.summarize_input.split(";")

                        try:
                            input_summarization = generate_text_summary_chunks(input_path_urls)

                            if input_summarization == "":
                                stream_response = generate_chat_completion_chunks(f"There is no content available for generating a summary.")
                                return EventSourceResponse(event_generator(stream_response))
                        
                            response_summarization = client.chat.completions.create(
                                                    model=payload.get("model", "generic"),
                                                    messages=input_summarization,
                                                    stream=True,
                                                    temperature=0.1,
                                                )
                            
                            return EventSourceResponse(event_generator(response_summarization))
                        
                        except Exception as e:
                            stream_response = generate_chat_completion_chunks(f"There was an error while creating the summary: {str(e)}")
                            return EventSourceResponse(event_generator(stream_response))
                        
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
                    
                if command == "/savetask":
                    if args[0] == "/file" or args[0] == "/web":
                        rag_persist_ok = context_manager.persist_task(args[1], args[0][1:])

                        if rag_persist_ok:
                            stream_response = generate_chat_completion_chunks(f"Persisting task was successful. Content can be reloaded with /loadtask {args[1]}.")
                            return EventSourceResponse(event_generator(stream_response))
                        else:
                            stream_response = generate_chat_completion_chunks("Persisting task was not successful. Please try again.")
                            return EventSourceResponse(event_generator(stream_response))

                    elif args[0] == "/summarize":
                        sum_persist_ok = context_manager.persist_task(args[1], "summarize")

                        if sum_persist_ok:
                            stream_response = generate_chat_completion_chunks(f"Persisting task was successful. Content can be reloaded with /loadtask {args[1]}.")
                            return EventSourceResponse(event_generator(stream_response))
                        else:
                            stream_response = generate_chat_completion_chunks("Persisting task was not successful. Please try again.")
                            return EventSourceResponse(event_generator(stream_response))
                    
                    else:
                        stream_response = generate_chat_completion_chunks("Usage: /savetask /<Task type> <Task name>\nPossible task types: /file, /web, /summarize")
                        return EventSourceResponse(event_generator(stream_response))
     
                if command == "/runtask":
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /runtask <Task name>")
                        return EventSourceResponse(event_generator(stream_response))

                    else:
                        task_load_res = context_manager.load_task(args[0])

                        if task_load_res == "db_err":
                            stream_response = generate_chat_completion_chunks("Loading task was not successful, the database does not exist or is write protected.")
                            return EventSourceResponse(event_generator(stream_response))
                        elif task_load_res == "not_existing":
                            stream_response = generate_chat_completion_chunks(f"Loading task was not successful because the task {args[0]} does not exist.")
                            return EventSourceResponse(event_generator(stream_response))
                        elif task_load_res == "error":
                            stream_response = generate_chat_completion_chunks("Loading task was not successful. Please try again.")
                            return EventSourceResponse(event_generator(stream_response))
                        
                        if task_load_res == "ok":
                            # Run the task according to task type
                            if context_manager.task_type == "web" or context_manager.task_type == "file":
                                # Vectorstore was automatically loaded
                                pass
                            elif context_manager.task_type == "summarize":
                                # Create summary
                                if not endpoint_avail():
                                    # No public OpenAI connection configured and local endpoint not available
                                    stream_response = generate_chat_completion_chunks("There is no LLM inference endpoint available. Please configure first and try again.")
                                    return EventSourceResponse(event_generator(stream_response))

                                input_path_urls = context_manager.summarize_input.split(";")

                                try:
                                    input_summarization = generate_text_summary_chunks(input_path_urls)

                                    if input_summarization == "":
                                        stream_response = generate_chat_completion_chunks(f"There is no content available for generating a summary.")
                                        return EventSourceResponse(event_generator(stream_response))
                                
                                    response_summarization = client.chat.completions.create(
                                                            model=payload.get("model", "generic"),
                                                            messages=input_summarization,
                                                            stream=True,
                                                            temperature=0.1,
                                                        )
                                    
                                    return EventSourceResponse(event_generator(response_summarization))
                                
                                except Exception as e:
                                    stream_response = generate_chat_completion_chunks(f"There was an error while creating the summary: {str(e)}")
                                    return EventSourceResponse(event_generator(stream_response))


                            stream_response = generate_chat_completion_chunks(f"Loading task {args[0]} was successful.")
                            return EventSourceResponse(event_generator(stream_response))

                if command == "/listtasks":
                    # List all tasks and output as markdown table
                    tasks = context_manager.list_all_tasks()
                    if not tasks:
                        table_md = "No tasks found."
                    else:
                        headers = [
                            "Task Name",
                            "RAG Content List",
                            "RAG Update Time",
                            "Summarize Input",
                            "Summarize Prompt",
                            "Task Type"
                        ]
                        table_md = "| " + " | ".join(headers) + " |\n"
                        table_md += "|---" * len(headers) + "|\n"
                        for t in tasks:
                            row = [
                                str(t.get("taskname", "")),
                                str(t.get("rag_content_list", "")),
                                str(t.get("rag_update_time", "")),
                                str(t.get("summarize_input", "")),
                                str(t.get("summarize_additional_prompt", "")),
                                str(t.get("task_type", ""))
                            ]
                            # Truncate long content for readability
                            row = [r if len(r) < 60 else r[:57] + "..." for r in row]
                            table_md += "| " + " | ".join(row) + " |\n"

                    stream_response = generate_chat_completion_chunks(table_md)
                    return EventSourceResponse(event_generator(stream_response))
                
                if command == "/taskinfo":
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /taskinfo <Task name>")
                        return EventSourceResponse(event_generator(stream_response))

                    else:
                        taskname = args[0]
                        task_info = context_manager.get_task_info(taskname)
                        if not task_info:
                            task_info_output = f"Task '{taskname}' not found."
                        else:
                            # Pretty, line-broken output with bold identifiers (markdown style)
                            fields = [
                                ("Task Name", str(task_info.get("taskname", ""))),
                                ("RAG Content List", str(task_info.get("rag_content_list", ""))),
                                ("RAG Update Time", str(task_info.get("rag_update_time", ""))),
                                ("Summarize Input", str(task_info.get("summarize_input", ""))),
                                ("Summarize Prompt", str(task_info.get("summarize_additional_prompt", ""))),
                                ("Task Type", str(task_info.get("task_type", ""))),
                            ]
                            # Truncate long content for readability
                            pretty_lines = []
                            for name, value in fields:
                                if len(value) > 200:
                                    value = value[:197] + "..."
                                pretty_lines.append(f"**{name}:** {value}")
                            task_info_output = "\n".join(pretty_lines)
                        stream_response = generate_chat_completion_chunks(task_info_output)
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/deletetask":
                    # Delete an existing task out of the database
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /deletetask <Task name>")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
                        taskname = args[0]
                        deleted = context_manager.delete_task(taskname)
                        if deleted:
                            stream_response = generate_chat_completion_chunks(f"Task '{taskname}' was deleted successfully.")
                        else:
                            stream_response = generate_chat_completion_chunks(f"Task '{taskname}' could not be deleted or does not exist.")
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
                
                if command == "/getconfigpaths":
                    # Outputs the paths of the config files
                    tmp_config_paths = []
                    tmp_config_paths.append(f"- **Chatshell config:** {self.chatshell_config_path}")
                    tmp_config_paths.append(f"- **Document path:** {self.doc_base_dir}")
                    tmp_config_paths.append(f"- **LLM server config:** {llm_server_config_path}")
                    tmp_config_paths.append(f"- **LLM model config:** {llm_config_path}")

                    stream_response = generate_chat_completion_chunks("\n".join(tmp_config_paths))
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
                    if rag_enabled:
                        tmp_rag.append(f"- **Mode:** {context_manager.task_type}")
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

                if command == "/refer":
                    # Refer to a specific part of the conversation
                    # Usage: /refer <msg_num|/lastmessage> <instruction>
                    if len(args) < 2:
                        stream_response = generate_chat_completion_chunks("Usage: /refer <msg_num|/lastmessage> <instruction>")
                        return EventSourceResponse(event_generator(stream_response))

                    msg_arg = args[0]
                    if msg_arg == "/lastmessage":
                        ref_message_dict = context_manager.get_lastmessage()
                        if not ref_message_dict:
                            stream_response = generate_chat_completion_chunks("No assistant message found to reference.")
                            return EventSourceResponse(event_generator(stream_response))
                    else:
                        try:
                            msg_num = int(msg_arg)
                        except Exception:
                            stream_response = generate_chat_completion_chunks("First argument must be an integer (e.g., -1 for last message) or /lastmessage")
                            return EventSourceResponse(event_generator(stream_response))

                        ref_message_dict = context_manager.get_message(msg_num)
                        if not ref_message_dict:
                            stream_response = generate_chat_completion_chunks(f"Message index {msg_num} is out of range.")
                            return EventSourceResponse(event_generator(stream_response))

                    # Join the rest as the instruction, remove surrounding quotes if present
                    instruction = " ".join(args[1:]).strip()
                    if instruction.startswith('"') and instruction.endswith('"'):
                        instruction = instruction[1:-1]

                    ref_message = ref_message_dict.get("content", "")

                    # Combine referenced message and instruction
                    combined_prompt = f"Message:\n{ref_message}\n\nInstruction:\n{instruction}"

                    # Prepare payload for completion
                    # Only use the referenced message and instruction as the new payload
                    payload["messages"] = [{"role": "user", "content": combined_prompt}]

                    # Streaming mode
                    if stream:
                        stream_response = client.chat.completions.create(**payload)
                        return EventSourceResponse(event_generator(stream_response))

                    # Non-streaming mode
                    response = client.chat.completions.create(**payload)

                    return JSONResponse(response.model_dump_json())

                if command == "/addinstruction":
                    if len(args) < 1:
                        stream_response = generate_chat_completion_chunks("Usage: /addinstruction <Instruction text>")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
                        instruction_content = " ".join(args).strip()
                        # Generate a unique random name for the instruction
                        instruction_name = f"user_{uuid.uuid4().hex[:8]}"
                        added = context_manager.save_instruction(instruction_name, instruction_content)
                        if added:
                            stream_response = generate_chat_completion_chunks(f"Instruction added with name: {instruction_name}")
                        else:
                            stream_response = generate_chat_completion_chunks("Failed to add instruction.")
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/saveinstruction":
                    if len(args) < 2:
                        stream_response = generate_chat_completion_chunks("Usage: /saveinstruction <Instruction name> <Instruction text>")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
                        instruction_name = args[0]
                        instruction_content = " ".join(args[1:]).strip()
                        added = context_manager.save_instruction(instruction_name, instruction_content)
                        if added:
                            stream_response = generate_chat_completion_chunks(f"Instruction saved with name: {instruction_name}")
                        else:
                            stream_response = generate_chat_completion_chunks("Failed to save instruction.")
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/loadinstruction":
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /loadinstruction <Instruction name>")
                        return EventSourceResponse(event_generator(stream_response))
                    else:
                        instruction_name = args[0]
                        loaded = context_manager.load_instruction(instruction_name)
                        if loaded:
                            # Find the loaded instruction in the list
                            for instr in context_manager.instruction_list:
                                if instr["instruction_name"] == instruction_name:
                                    content = instr["instruction_content"]
                                    break
                            else:
                                content = "Instruction loaded but not found in list."
                            stream_response = generate_chat_completion_chunks(f"Instruction '{instruction_name}':\n{content}")
                        else:
                            stream_response = generate_chat_completion_chunks(f"Failed to load instruction '{instruction_name}'.")
                        return EventSourceResponse(event_generator(stream_response))

                if command == "/runinstruction":
                    # Load instruction vom DB and invoke inference
                    reference_message = None

                    if "/lastmessage" in args:
                        args_wo_flag = [a for a in args if a != "/lastmessage"]
                        if len(args_wo_flag) != 1:
                            stream_response = generate_chat_completion_chunks("Usage: /runinstruction <Instruction name> [/lastmessage | /refer <msg_num>]")
                            return EventSourceResponse(event_generator(stream_response))
                        instruction_name = args_wo_flag[0]
                        reference_message = context_manager.get_lastmessage()
                        if not reference_message:
                            stream_response = generate_chat_completion_chunks("No assistant message found to reference.")
                            return EventSourceResponse(event_generator(stream_response))
                    elif "/refer" in args:
                        try:
                            refer_pos = args.index("/refer")
                            msg_arg = args[refer_pos + 1]
                            msg_num = int(msg_arg)
                            if refer_pos != 1 or len(args) != 3:
                                raise ValueError()
                            instruction_name = args[0]
                        except (ValueError, IndexError):
                            stream_response = generate_chat_completion_chunks("Usage: /runinstruction <Instruction name> /refer <msg_num>")
                            return EventSourceResponse(event_generator(stream_response))

                        reference_message = context_manager.get_message(msg_num)
                        if not reference_message:
                            stream_response = generate_chat_completion_chunks(f"Message index {msg_num} is out of range.")
                            return EventSourceResponse(event_generator(stream_response))
                    else:
                        if len(args) != 1:
                            stream_response = generate_chat_completion_chunks("Usage: /runinstruction <Instruction name> [/lastmessage | /refer <msg_num>]")
                            return EventSourceResponse(event_generator(stream_response))
                        instruction_name = args[0]

                    instruction_content = context_manager.load_instruction(instruction_name)
                    if not instruction_content:
                        stream_response = generate_chat_completion_chunks(f"Failed to load instruction '{instruction_name}'.")
                        return EventSourceResponse(event_generator(stream_response))

                    # Prepare new payload messages
                    if reference_message is not None:
                        ref_message_content = reference_message.get("content", "")
                        combined_prompt = f"Message:\n{ref_message_content}\n\nInstruction:\n{instruction_content}"
                        payload["messages"] = [{"role": "user", "content": combined_prompt}]
                    else:
                        # Default: use all messages, append instruction as last user message
                        payload["messages"].append({"role": "user", "content": instruction_content})

                    # Streaming mode
                    if stream:
                        stream_response = client.chat.completions.create(**payload)
                        return EventSourceResponse(event_generator(stream_response))
                    
                    # Non-streaming mode
                    response = client.chat.completions.create(**payload)
                    return JSONResponse(response.model_dump_json())

                if command == "/save":
                    # Save a conversation to output folder
                    if len(args) != 1:
                        stream_response = generate_chat_completion_chunks("Usage: /save <format json or md>")
                        return EventSourceResponse(event_generator(stream_response))

                    else:
                        # Save the entire conversation in payload, removing the last user message
                        messages = payload.get("messages", [])
                        if messages:
                            messages_to_save = messages[:-1]  # Remove last user message
                        else:
                            messages_to_save = []

                        # Determine format
                        fmt = args[0].lower()

                        filename = f"conversation_{uuid.uuid4().hex[:8]}"
                        output_dir = self.output_base_dir
                        output_dir.mkdir(parents=True, exist_ok=True)

                        if fmt == "json":
                            filepath = output_dir / f"{filename}.json"
                            with open(filepath, "w", encoding="utf-8") as f:
                                json.dump(messages_to_save, f, ensure_ascii=False, indent=2)

                            stream_response = generate_chat_completion_chunks(f"Conversation saved as {filepath}")
                            return EventSourceResponse(event_generator(stream_response))

                        elif fmt == "md":
                            filepath = output_dir / f"{filename}.md"
                            # Convert messages to markdown
                            md_lines = []
                            for msg in messages_to_save:
                                role = msg.get("role", "user")
                                content = msg.get("content", "")
                                md_lines.append(f"**{role.capitalize()}:**\n{content}\n\n---\n")
                            with open(filepath, "w", encoding="utf-8") as f:
                                f.writelines(md_lines)

                            stream_response = generate_chat_completion_chunks(f"Conversation saved as {filepath}")
                            return EventSourceResponse(event_generator(stream_response))

                        else:
                            stream_response = generate_chat_completion_chunks("Format must be 'json' or 'md'.")
                            return EventSourceResponse(event_generator(stream_response))

                if command == "/shellmode":
                    # Activate shell mode for specific chat by inserting the keyword
                    stream_response = generate_chat_completion_chunks(f"This chat is now marked as shell-chat, no LLM interaction will be performed on future inputs.")
                    return EventSourceResponse(event_generator(stream_response))

                if command == "/version":
                    # Output application version
                    stream_response = generate_chat_completion_chunks(f"Chatshell application version: {self.version}")
                    return EventSourceResponse(event_generator(stream_response))
                
                if command == "/exit":
                    # Properly quit chatshell server
                    try:
                        llm_server.stop_all_processes()
                    except Exception as e:
                        print(f"Error stopping LLM server processes: {e}")
                    # Attempt to gracefully shutdown the server if possible
                    try:
                        if 'server' in locals() and hasattr(server, 'should_exit'):
                            server.should_exit = True
                    except Exception as e:
                        print(f"Error signaling server shutdown: {e}")
                    stream_response = generate_chat_completion_chunks("Chatshell server is shutting down.")
                    return EventSourceResponse(event_generator(stream_response))
                
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
                
                # Add instruction messages
                instructions = context_manager.get_instructions()
                # Convert each instruction to an OpenAI chat message (role: 'user')
                instruction_messages = [
                    {"role": "user", "content": instr}
                    for instr in instructions if instr.strip() != ""
                ]
                # Add all instruction messages as user messages to payload
                payload["messages"] = payload["messages"] + instruction_messages
                
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
