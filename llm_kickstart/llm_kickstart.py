import multiprocessing
import subprocess
import os
import signal
import time
import json
import appdirs
import sys
from pathlib import Path


class LLMKickstart:
    def __init__(self):
        CONFIG_DIR = Path(appdirs.user_config_dir(appname='LLM_Kickstart'))
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        self.llm_config_path        = CONFIG_DIR / 'llm_config.json'
        self.app_config_path        = CONFIG_DIR / 'app_config.json'
        self.proc_list              = CONFIG_DIR / 'process_list.json'

        self.target_server_app      = ""
        self.use_python_server_lib  = False

        self.llm_config             = None
        self.app_config             = None
        self.load_config(llm_config_path=self.llm_config_path, app_config_path=self.app_config_path)

        self.processes              = {}
        self.output_cache           = ""  # Cache for all process output

        self.update_process_list_file()

    def load_config(self, llm_config_path="llm_config.json", app_config_path="app_config.json"):
        """
        Load and parse the llm_config.json file into structured variables.
        """
        try:
            if not self.llm_config_path.exists():
                # Create llm config file if not existing
                tmp_llm_config = [
                    {
                        "name": "Local_LLM_Model",
                        "ip": "",
                        "port": "4000",
                        "model": "llm_model.gguf",
                        "ctx-size": "",
                        "flash-attn": "",
                        "no-kv-offload": "",
                        "no-mmap": "",
                        "cache-type-k": "",
                        "cache-type-v": "",
                        "n-gpu-layers": "",
                        "lora": "",
                        "no-context-shift": "",
                        "api-key": ""
                    }
                ]

                with self.llm_config_path.open('w') as f:
                    json.dump(tmp_llm_config, f, indent=4)
                    
            with open(llm_config_path, "r") as f:
                self.llm_config = json.load(f)
        except Exception as e:
            print(f"Failed to load config file {llm_config_path}: {e}")
            self.llm_config = None

        """
        Load and parse the app_config.json file into structured variables.
        """
        try:
            if not self.app_config_path.exists():
                # Create llm config file if not existing
                tmp_app_config = {
                    "llama-server-path": "/Users/Julian/Downloads/llm_models_gguf/llama.cpp/build/bin/llama-server",
                    "use-llama-server-python": "False"
                    }

                with self.app_config_path.open('w') as f:
                    json.dump(tmp_app_config, f, indent=4)

            with open(app_config_path, "r") as f:
                self.app_config = json.load(f)
                self.target_server_app = self.app_config["llama-server-path"]
                self.use_python_server_lib = json.loads(str(self.app_config["use-llama-server-python"]).lower())

                # Check app file if python lib is not activated
                if not self.use_python_server_lib:
                    if os.path.exists(self.target_server_app ):
                        print("Error: llama-server executable file not found.")

        except Exception as e:
            print(f"Failed to load config file {app_config_path}: {e}")
            self.app_config = None

    def get_llm_config(self):
        return self.llm_config_path, self.llm_config

    def get_app_config(self):
        return self.app_config_path, self.app_config

    def refresh_config(self):
        self.llm_config = None
        self.app_config = None
        self.load_config(llm_config_path=self.llm_config_path, app_config_path=self.app_config_path)
    
    def create_endpoint(self, name):
        """
        Start a new process running ./llama_server with parameters from the config for the given LLM name.
        """
        if self.llm_config is None:
            print("Configuration not loaded. Please call load_config() first.")
            return

        # Find the LLM config by name
        llm_config = None
        for conf in self.llm_config:
            if conf.get("name") == name:
                llm_config = conf
                break

        if llm_config is None:
            print(f"No configuration found for LLM with name '{name}'.")
            return

        # Build command line arguments from the config
        args = []
        self.args_dict = {}
        for key, value in llm_config.items():
            if key == "name":
                continue  # skip name in args

            if value == "" or str(value).lower() == "default":
                continue # skip default values

            # Convert key to command line argument format, e.g. "model-path" -> "--model-path"
            arg_key = f"--{key}"

            # Convert boolean to flag or no flag
            if str(value).lower() == "true" or str(value).lower() == "false":
                if value:
                    args.append(arg_key)
                    self.args_dict[arg_key] = True
                # if false, skip adding the flag
            else:
                args.append(arg_key)
                args.append(str(value))
                self.args_dict[arg_key] = value

        # Start the process using create_process
        self.create_process(name, self.target_server_app, *args)

    def create_process(self, name, executable_path, *args):
        if name in self.processes:
            print(f"Process with name '{name}' already exists.")
            return

        # Start the process in a new process group so we can kill all children
        if self.use_python_server_lib:
            # Use python bindings instead of binaries
            process = subprocess.Popen([
                                sys.executable,
                                "-m", "llama_cpp.server",
                                "--model", self.args_dict["--model"],
                                "--port", self.args_dict["--port"],
                            ], preexec_fn=os.setsid)
            self.processes[name] = process

        else:
            # Check app file if python lib is not activated
            if not os.path.exists(self.target_server_app ):
                print("Error: llama-server executable file not found.")
                return

            process = subprocess.Popen([executable_path, *args], preexec_fn=os.setsid)
            self.processes[name] = process

        print(f"Process '{name}' started with PID {process.pid}.")
        self.update_process_list_file()

    def stop_process(self, name):
        if name not in self.processes:
            print(f"No process found with name '{name}'.")
            return

        process = self.processes[name]
        if process.poll() is None:
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
                print(f"Process '{name}' with PID {process.pid} has been stopped.")
            except Exception as e:
                print(f"Failed to kill process group for '{name}': {e}")
        else:
            print(f"Process '{name}' is not running.")
        del self.processes[name]
        self.update_process_list_file()

    def restart_process(self, name, target, *args):
        self.stop_process(name)
        self.create_endpoint(name)
        self.update_process_list_file()

    def stop_all_processes(self):
        names = list(self.processes.keys())
        for name in names:
            self.stop_process(name)
        self.update_process_list_file()
    
    def list_processes(self):
        if len(self.processes.items()) > 0:
            for name, process in self.processes.items():
                status = "running" if process.poll() is None else "stopped"
                print(f"- Process '{name}': PID {process.pid}, Status: {status}")
        else:
            print("- no processes currently running -")

        self.update_process_list_file()

    def update_process_list_file(self):
        process_list = {
            name: {
                "pid": process.pid,
                "status": "running" if process.poll() is None else "stopped"
            }
            for name, process in self.processes.items()
        }
        with open(self.proc_list, "w") as file:
            json.dump(process_list, file, indent=4)
            