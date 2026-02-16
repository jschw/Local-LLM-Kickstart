import os
import appdirs
from datetime import datetime
from pathlib import Path
import sqlite3
import json
import pyperclip
from urllib.parse import urlparse
from PyPDF2 import PdfReader
from .utils_rag import crawl_website

class ContextManager:
    """
    Context manager for handling RAG context, RAG provider, and prompt management.
    """
    def __init__(self):
        from .vectorstore import ChatshellVectorsearch
        self.doc_base_dir = None

        DB_DIR = Path(appdirs.user_config_dir(appname='chatshell'))
        self.context_manager_db_path  = DB_DIR / 'context_manager.sqlite'

        self.task_type          = ""

        self.context_list       = []
        self.context_active     = False
        self.context_update_time = ""

        self.rag_provider       = ChatshellVectorsearch()
        self.rag_active         = False
        self.rag_content_list   = []
        self.rag_update_time    = ""

        self.summarize_input    = ""
        self.summarize_text     = ""
        self.summarize_additional_prompt = ""

        # Instruction management
        self.instruction_list        = []

        # Current payload
        self.payload            = None

    def _get_payload_messages(self):
        """
        Return the messages list from the current payload or an empty list if unavailable.
        """
        if not isinstance(self.payload, dict):
            return []
        messages = self.payload.get("messages", [])
        return messages if isinstance(messages, list) else []

    def get_lastmessage(self):
        """
        Retrieve the last assistant message preceding the latest user entry.
        Returns the message dictionary or None if no such message exists.
        """
        messages = self._get_payload_messages()
        if not messages:
            return None

        start_index = len(messages) - 1
        if start_index >= 0 and messages[start_index].get("role") == "user":
            start_index -= 1

        for index in range(start_index, -1, -1):
            message = messages[index]
            if message.get("role") == "assistant":
                return message

        return None

    def get_message(self, msg_num):
        """
        Retrieve a message by its absolute or negative index. Returns the
        message dictionary or None if the index is out of bounds.
        """
        messages = self._get_payload_messages()
        if not messages:
            return None

        if not isinstance(msg_num, int):
            return None

        index = msg_num if msg_num >= 0 else len(messages) + msg_num
        if 0 <= index < len(messages):
            return messages[index]

        return None

    def save_instruction(self, instruction_name, instruction_content):
        """
        Save an instruction to the database if both name and content are provided.
        Updates self.instruction_list by adding or updating the instruction.
        """
        if not instruction_name or not instruction_content:
            print("Instruction name and content must be provided.")
            return False
        db_path = str(self.context_manager_db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS instructions (
                    instruction_name TEXT PRIMARY KEY,
                    instruction_content TEXT
                )
            """)
            c.execute("""
                INSERT INTO instructions (instruction_name, instruction_content)
                VALUES (?, ?)
                ON CONFLICT(instruction_name) DO UPDATE SET
                    instruction_content=excluded.instruction_content
            """, (instruction_name, instruction_content))
            conn.commit()
            return True

        except Exception as e:
            print(f"Error saving instruction: {e}")
            return False
        finally:
            conn.close()

    def load_instruction(self, instruction_name):
        """
        Load an instruction by name from the database. Returns the instruction content or None.
        Adds the loaded instruction to self.instruction_list if found and not already present.
        """
        db_path = str(self.context_manager_db_path)
        if not os.path.exists(db_path):
            print("DB not existing.")
            return False
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            c.execute("""
                SELECT instruction_content FROM instructions WHERE instruction_name = ?
            """, (instruction_name,))
            row = c.fetchone()
            if row is not None:
                # Only add if not already in instruction_list
                if not any(i["instruction_name"] == instruction_name for i in self.instruction_list):
                    self.instruction_list.append({"instruction_name": instruction_name, "instruction_content": row[0]})
                return row[0]
            else:
                print(f"No instruction found for: {instruction_name}")
                return False
        except Exception as e:
            print(f"Error loading instruction: {e}")
            return False
        finally:
            conn.close()

    def list_all_instructions(self):
        """
        Returns a list of all instructions stored in the database. Each instruction is a dictionary with 'instruction_name' and 'instruction_content'.
        """
        db_path = str(self.context_manager_db_path)
        if not os.path.exists(db_path):
            print("DB not existing.")
            return []
        conn = sqlite3.connect(db_path)
        instructions = []
        try:
            c = conn.cursor()
            c.execute("SELECT instruction_name, instruction_content FROM instructions")
            rows = c.fetchall()
            for row in rows:
                instructions.append({"instruction_name": row[0], "instruction_content": row[1]})
        except Exception as e:
            print(f"Error listing all instructions: {e}")
            return []
        finally:
            conn.close()
        return instructions

    def get_instructions(self):
        """
        Returns a list of the content of all existing instructions in self.instruction_list.
        """
        return [instr["instruction_content"] for instr in self.instruction_list]

    def delete_instruction(self, instruction_name):
        """
        Delete an instruction by name from the database and remove it from self.instruction_list.
        """
        db_path = str(self.context_manager_db_path)
        if not os.path.exists(db_path):
            print("DB not existing.")
            return False
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            c.execute("DELETE FROM instructions WHERE instruction_name = ?", (instruction_name,))
            deleted = c.rowcount > 0
            conn.commit()
            # Remove from list if present
            self.instruction_list = [i for i in self.instruction_list if i["instruction_name"] != instruction_name]
            return deleted
        except Exception as e:
            print(f"Error deleting instruction: {e}")
            return False
        finally:
            conn.close()

    """
    Context manager for handling RAG context and RAG provider.
    """
    def __init__(self):
        from .vectorstore import ChatshellVectorsearch
        self.doc_base_dir = None

        DB_DIR = Path(appdirs.user_config_dir(appname='chatshell'))
        self.context_manager_db_path  = DB_DIR / 'context_manager.sqlite'

        self.task_type          = ""

        self.context_list       = []
        self.context_active     = False
        self.context_update_time = ""

        self.rag_provider       = ChatshellVectorsearch()
        self.rag_active         = False
        self.rag_content_list   = []
        self.rag_update_time    = ""

        self.summarize_input    = ""
        self.summarize_text     = ""
        self.summarize_additional_prompt = ""

    def get_text_clipboard(self):
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
        
    def is_url(self, path_or_url):
            """
            Returns True if the input is an HTTP/HTTPS URL, False if it's a file path.
            """
            try:
                result = urlparse(path_or_url)
                return result.scheme in ("http", "https")
            except Exception:
                return False
    
    def set_doc_base_dir(self, doc_base_dir):
        self.doc_base_dir = doc_base_dir

    def rag_update_file(self, document_path) -> list:
        # Split paths if more than one
        document_paths_arg = document_path.split(";")

        # Check document exist
        document_paths_exist = []
        output_list = []

        for doc in document_paths_arg:
            doc_current = doc
            if not os.path.isfile(doc_current):
                # Document is not available at absolute path, checking rel. path
                doc_current = os.path.join(self.doc_base_dir, doc_current)
                if not os.path.isfile(doc_current):
                    # Document is not available -> return error
                    print(f"--> Document {doc_current} not found.")
                    output_list.append(f"Document {doc_current} not found.")
                    continue

            output_list.append(f"Document {doc_current} existing and added to RAG document list.")
            document_paths_exist.append(doc_current)

        if len(document_paths_exist) == 0:
            print("--> No existing document found at given path.")
            output_list.insert(0, "No existing document found at given path.")
            return [False, "\n".join(output_list)]

        # Update RAG
        self.rag_content_list = document_paths_exist
        self.rag_update_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        self.task_type = "file"
        self.rag_active = True

        rag_update_ok = self.rag_provider.init_vectorstore_pdf(document_paths_exist)

        if rag_update_ok:
            output_list.insert(0, "Ready, you can now chat with your document(s)!")
        else:
            output_list.insert(0, "There was a problem updating the RAG system. Please try again.")

        return [rag_update_ok, "\n".join(output_list)]

    def rag_update_directory(self, path, filetype=None):
        """
        Update the RAG vectorstore with all files in a directory (optionally filtered by filetype).
        Supports: pdf, txt, json, md. If filetype is None, all supported files are included.
        """
        output_list = []
        directory = Path(path)
        if not directory.is_dir():
            output_list.append(f"Directory {path} not found.")
            return [False, "\n".join(output_list)]

        supported_types = {"pdf", "txt", "json", "md"}
        def get_ext(f):
            return Path(f).suffix.lower().lstrip('.')

        # Gather files
        files = [str(f) for f in directory.rglob("*") if f.is_file()]
        if filetype is None:
            filtered_files = [f for f in files if get_ext(f) in supported_types]
        else:
            ext = filetype.lower().lstrip('.')
            if ext not in supported_types:
                output_list.append(f"Filetype '{filetype}' is not supported. Supported: {', '.join(supported_types)}.")
                return [False, "\n".join(output_list)]
            filtered_files = [f for f in files if get_ext(f) == ext]

        if not filtered_files:
            output_list.append(f"No supported files found in {path} matching type '{filetype}'.")
            return [False, "\n".join(output_list)]

        self.rag_content_list = filtered_files
        self.rag_update_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        self.task_type = "directory"
        self.rag_active = True

        # Only allow all-pdf or all-txt/json/md (no mixing)
        if all(get_ext(f) == "pdf" for f in filtered_files):
            rag_update_ok = self.rag_provider.init_vectorstore_pdf(filtered_files)
        elif all(get_ext(f) in {"txt", "json", "md"} for f in filtered_files):
            text_content = ""
            for f in filtered_files:
                try:
                    with open(f, "r", encoding="utf-8") as fin:
                        text_content += fin.read() + "\n"
                except Exception as e:
                    output_list.append(f"Error reading {f}: {e}")
            if not text_content.strip():
                output_list.append("No text content could be read from the files.")
                return [False, "\n".join(output_list)]
            rag_update_ok = self.rag_provider.init_vectorstore_str(text_content)
        else:
            output_list.append("Mixed or unsupported file types in selection. Only all-pdf or all-txt/json/md supported.")
            return [False, "\n".join(output_list)]

        if rag_update_ok:
            output_list.insert(0, "Ready, you can now chat with your document(s)!")
        else:
            output_list.insert(0, "There was a problem updating the RAG system. Please try again.")

        return [rag_update_ok, "\n".join(output_list)]


    def rag_update_web(self, url, deep):
        # Split paths if more than one
        urls = url.split(";")
        self.rag_update_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')

        # Update RAG
        self.rag_content_list = urls
        self.task_type = "web"
        self.rag_active = True
        rag_update_ok = self.rag_provider.init_vectorstore_web(urls, deep)

        return rag_update_ok
    
    def rag_update_clipbrd(self, input):
        rag_update_ok = self.rag_provider.init_vectorstore_str(input)
        self.task_type = "clip"

        return rag_update_ok

    def __enter__(self):
        # Optionally, could accept initial context here
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset_context()

    def add_context(self, input):
        self.task_type = "context"
        self.context_active = True
        self.context_update_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        self.context_list.append(input)

    def get(self):
        context = ""
        for i in self.context_list:
            context += f"{{\n{i}\n}},\n"
        return context

    def reset_context(self):
        self.context_list = []
        self.context_active = False

    def reset_rag_context(self):
        self.rag_content_list = []
        self.rag_active = False

    def summarize_files_urls(self, input_path_url)->list:
        chunk_list = []

        # Check if argument is clipboard content
        if "/clipboard" in input_path_url:
            # Handle as Clipboard content
            clip_content = self.get_text_clipboard()
            if clip_content != None:
                chunk_list = [clip_content]
            else:
                return ["The clipboard is empty or not valid text content.", False]

        # Use is_url to check if input_path_url is a URL or a file path
        elif self.is_url(input_path_url):
            # Handle as URL
            # Crawl website
            print(f"-> Crawling {input_path_url}.")
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
                    return [f"The document {input_path_url} was not found.\nPlease enter a valid document path.", False]

            # --> Read PDF pages into chunk list
            reader = PdfReader(doc_current)

            # Load each page's text into a list, one entry per page
            for page in reader.pages:
                text = page.extract_text()
                chunk_list.append(text)

        try:
            # Create summary
            print("--> Start summarization...")
            text_summary = self.rag_provider.generate_text_summary(chunk_list)
            print("--> Generated summary chunks.")

            self.summarize_text = text_summary
            self.task_type_summary = True

            return [text_summary, True]

        except Exception as e:
            return [f"There was an error while creating the summary: {str(e)}", False]

    def persist_task(self, taskname, tasktype):
        """
        Persists RAG or summarize task data into a unified SQLite 'tasks' table.
        The table includes columns for both RAG and summarize fields.
        'mode' should be either 'rag' or 'summarize' to determine which fields to update.
        """
        db_path = str(self.context_manager_db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    taskname TEXT PRIMARY KEY,
                    rag_content_list TEXT,
                    rag_update_time TEXT,
                    summarize_input TEXT,
                    summarize_additional_prompt TEXT,
                    task_type TEXT
                )
            """)
            if tasktype == "file" or tasktype == "web":
                rag_content_json = json.dumps(self.rag_content_list)
                c.execute("""
                    INSERT INTO tasks (taskname, rag_content_list, rag_update_time, task_type)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(taskname) DO UPDATE SET
                        rag_content_list=excluded.rag_content_list,
                        rag_update_time=excluded.rag_update_time,
                        task_type=excluded.task_type
                """, (taskname, rag_content_json, self.rag_update_time, tasktype))
            elif tasktype == "summarize":
                print(f"Summarize input: {self.summarize_input}")
                c.execute("""
                    INSERT INTO tasks (taskname, summarize_input, summarize_additional_prompt, task_type)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(taskname) DO UPDATE SET
                        summarize_input=excluded.summarize_input,
                        summarize_additional_prompt=excluded.summarize_additional_prompt,
                        task_type=excluded.task_type
                """, (taskname, self.summarize_input, self.summarize_additional_prompt, tasktype))
            else:
                return False
            conn.commit()
        except Exception:
            return False
        finally:
            conn.close()
        return True

    def load_task(self, taskname):
        """
        Retrieves all task data for a given taskname from the SQLite database.
        Returns a dictionary with all available fields if found, otherwise None.
        """
        db_path = str(self.context_manager_db_path)
        print(db_path)
        if not os.path.exists(db_path):
            print("DB not existing.")
            return "db_err"
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            c.execute(
                "SELECT rag_content_list, rag_update_time, summarize_input, summarize_additional_prompt, task_type FROM tasks WHERE taskname = ?",
                (taskname,)
            )
            row = c.fetchone()
            print(row)
            if row is not None:
                # Write all values to local variables
                self.rag_content_list               = list(json.loads(row[0])) if row[0] else None
                self.rag_update_time                = row[1]
                self.summarize_input                = row[2]
                self.summarize_additional_prompt    = row[3]
                self.task_type                      = row[4]

                if self.task_type == "file":
                    # Run vectorstore update for file(s)
                    self.rag_update_file(self.rag_content_list)

                elif self.task_type == "web":
                    # Run vectorstore update for URL(s)
                    self.rag_update_web(self.rag_content_list, False)

                result = {
                    "rag_content_list": self.rag_content_list,
                    "rag_update_time": self.rag_update_time,
                    "summarize_input": self.summarize_input,
                    "summarize_additional_prompt": self.summarize_additional_prompt,
                    "task_type": self.task_type
                }

                return "ok"
            
            print("No entry found.")
            return "not_existing"
        except Exception as e:
            print(str(e))
            return "error"
        finally:
            conn.close()

    def list_all_tasks(self):
        """
        Returns a list of all tasks with their values from the SQLite database.
        Each task is represented as a dictionary.
        """
        db_path = str(self.context_manager_db_path)
        if not os.path.exists(db_path):
            print("DB not existing.")
            return []
        conn = sqlite3.connect(db_path)
        tasks = []
        try:
            c = conn.cursor()
            c.execute(
                "SELECT taskname, rag_content_list, rag_update_time, summarize_input, summarize_additional_prompt, task_type FROM tasks"
            )
            rows = c.fetchall()
            for row in rows:
                task = {
                    "taskname": row[0],
                    "rag_content_list": json.loads(row[1]) if row[1] else None,
                    "rag_update_time": row[2],
                    "summarize_input": row[3],
                    "summarize_additional_prompt": row[4],
                    "task_type": row[5]
                }
                tasks.append(task)
        except Exception as e:
            print(f"Error listing all tasks: {e}")
            return []
        finally:
            conn.close()
        return tasks

    def delete_task(self, taskname):
        """
        Deletes the task with the given taskname from the SQLite database.
        Returns True if a row was deleted, False otherwise.
        """
        db_path = str(self.context_manager_db_path)
        if not os.path.exists(db_path):
            print("DB not existing.")
            return False
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            c.execute("DELETE FROM tasks WHERE taskname = ?", (taskname,))
            deleted = c.rowcount > 0
            conn.commit()
            return deleted
        except Exception as e:
            print(f"Error deleting task: {e}")
            return False
        finally:
            conn.close()

    def get_task_info(self, taskname):
        """
        Returns all info for the specific task with its values as a dictionary.
        If the task does not exist, returns None.
        """
        db_path = str(self.context_manager_db_path)
        if not os.path.exists(db_path):
            print("DB not existing.")
            return None
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            c.execute(
                "SELECT taskname, rag_content_list, rag_update_time, summarize_input, summarize_additional_prompt, task_type FROM tasks WHERE taskname = ?",
                (taskname,)
            )
            row = c.fetchone()
            if row is not None:
                task_info = {
                    "taskname": row[0],
                    "rag_content_list": json.loads(row[1]) if row[1] else None,
                    "rag_update_time": row[2],
                    "summarize_input": row[3],
                    "summarize_additional_prompt": row[4],
                    "task_type": row[5]
                }
                return task_info
            else:
                print("No entry found for task:", taskname)
                return None
        except Exception as e:
            print(f"Error retrieving task info: {e}")
            return None
        finally:
            conn.close()
