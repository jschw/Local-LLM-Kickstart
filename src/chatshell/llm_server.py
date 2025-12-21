import subprocess
import os
import signal
import json
import appdirs
import sys
from pathlib import Path


class LocalLLMServer:
    def __init__(self, termux_paths=False):
        self.termux = termux_paths

        CONFIG_DIR = Path(appdirs.user_config_dir(appname='chatshell'))
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        self.llm_config_path        = CONFIG_DIR / 'llm_config.json'
        self.llm_server_config_path = CONFIG_DIR / 'llm_server_config.json'
        self.proc_list              = CONFIG_DIR / 'process_list.json'

        self.target_server_app      = ""
        self.use_python_server_lib  = False

        if self.termux:
            self.model_base_dir         = Path(os.path.expanduser("~/storage/shared/chatshell/Models"))
        else:
            self.model_base_dir         = Path(os.path.expanduser("~/chatshell/Models"))

        self.model_base_dir.mkdir(parents=True, exist_ok=True)

        self.llm_config             = None
        self.llm_server_config      = None
        self.load_config()

        self.processes              = {}
        self.output_cache           = ""  # Cache for all process output

        self.update_process_list_file()

    def load_config(self):
        """
        Load and parse the llm_config.json file into structured variables.
        """
        try:
            if not self.llm_config_path.exists():
                # Create llm config file if not existing
                # Template content of llm_config.json
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
                    
            with open(self.llm_config_path, "r") as f:
                self.llm_config = json.load(f)
        except Exception as e:
            print(f"Failed to load config file {self.llm_config_path}: {e}")
            self.llm_config = None

        """
        Load and parse the llm_server_config.json file into structured variables.
        """
        try:
            if not self.llm_server_config_path.exists():
                # Create llm config file if not existing
                # Template content of the llm_server_config.json
                tmp_llm_server_config = {
                    "llama-server-path": "~/llama.cpp/build/bin/llama-server",
                    "use-llama-server-python": "True"
                    }

                with self.llm_server_config_path.open('w') as f:
                    json.dump(tmp_llm_server_config, f, indent=4)

            with open(self.llm_server_config_path, "r") as f:
                self.llm_server_config = json.load(f)
                self.target_server_app = Path(os.path.expanduser(self.llm_server_config["llama-server-path"]))
                self.use_python_server_lib = json.loads(str(self.llm_server_config["use-llama-server-python"]).lower())

                # Check app file if python lib is not activated
                if not self.use_python_server_lib:
                    if not os.path.exists(self.target_server_app ):
                        print("--> Error: llama-server executable file not found.")

        except Exception as e:
            print(f"Failed to load config file {self.llm_server_config_path}: {e}")
            self.llm_server_config = None

    def show_llm_config(self, config_set):
        found = False
        for conf in self.llm_config:
            if conf.get("name") == config_set:
                print(f"Parameters for LLM config set '{config_set}':")
                for k, v in conf.items():
                    print(f"  {k}: {v}")
                found = True
                break
        if not found:
            print(f"No LLM config set found with name '{config_set}'.")

    def show_llm_server_config(self):
        print("Parameters in llm_server_config.json:")
        for k, v in self.llm_server_config.items():
            print(f"  {k}: {v}")
    
    def get_llm_server_config_path(self):
        return self.llm_server_config_path
    
    def get_llm_server_config(self):
        return self.llm_server_config_path, self.llm_server_config

    def get_llm_config_path(self):
        return self.llm_config_path
    
    def get_llm_config(self):
        return self.llm_config_path, self.llm_config

    def refresh_config(self):
        self.llm_config = None
        self.llm_server_config = None
        self.load_config(llm_config_path=self.llm_config_path, llm_server_config_path=self.llm_server_config_path)

    def listendpoints(self):
        print("Available LLM endpoint configs:")

        for config in self.llm_config:
            name = config.get("name", "Unnamed LLM")
            print(f"  - {name}")

    def edit_llm_conf(self, config_set, key, value):
        found = False
        for conf in self.llm_config:
            if conf.get("name") == config_set:
                conf[key] = value
                found = True
                break
        if not found:
            print(f"No LLM config set found with name '{config_set}'.")
        else:
            try:
                with open(self.llm_config_path, "w") as f:
                    json.dump(self.llm_config, f, indent=4)
                self.refresh_config()
                print(f"Updated '{key}' in LLM config set '{config_set}' to '{value}'.")
            except Exception as e:
                print(f"Failed to update llm_config.json: {e}")

    def edit_llm_server_conf(self, key, value):
        if key not in self.llm_server_config:
            print(f"Key '{key}' not found in llm_server_config.json. Adding it.")
        self.llm_server_config[key] = value
        try:
            with open(self.llm_server_config_path, "w") as f:
                json.dump(self.llm_server_config, f, indent=4)
            self.refresh_config()
            print(f"Updated '{key}' in llm_server_config.json to '{value}'.")
        except Exception as e:
            print(f"Failed to update llm_server_config.json: {e}")

    def create_new_llm_config(self, new_name):
        # Check for duplicate
        if any(conf.get("name") == new_name for conf in self.llm_config):
            print(f"LLM config set with name '{new_name}' already exists.")
        else:
            # Use the template from llm_kickstart.py
            template = {
                "name": new_name,
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
            self.llm_config.append(template)
            try:
                with open(self.llm_config_path, "w") as f:
                    json.dump(self.llm_config_path, f, indent=4)
                self.refresh_config()
                print(f"Created new LLM config set '{new_name}'.")
            except Exception as e:
                print(f"Failed to create new LLM config set: {e}")

    def delete_llm_config(self, del_name):
        found = False
        for i, conf in enumerate(self.llm_config):
            if conf.get("name") == del_name:
                del self.llm_config[i]
                found = True
                break
        if not found:
            print(f"No LLM config set found with name '{del_name}'.")
        else:
            try:
                with open(self.llm_config_path, "w") as f:
                    json.dump(self.llm_config, f, indent=4)
                self.refresh_config()
                print(f"Deleted LLM config set '{del_name}'.")
            except Exception as e:
                print(f"Failed to delete LLM config set: {e}")
    
    def rename_llm_config(self, old_name, new_name):
        found = False
        for conf in self.llm_config:
            if conf.get("name") == old_name:
                conf["name"] = new_name
                found = True
                break
        if not found:
            print(f"No LLM config set found with name '{old_name}'.")
        else:
            try:
                with open(self.llm_config_path, "w") as f:
                    json.dump(self.llm_config, f, indent=4)
                self.refresh_config()
                print(f"Renamed LLM config set from '{old_name}' to '{new_name}'.")
            except Exception as e:
                print(f"Failed to rename LLM config set: {e}")
    
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
            print(f"--> No configuration found for LLM with name '{name}'.")
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
                if arg_key == "--model":
                    # Check if model is at absolute path or model base dir available
                    model_path = str(value)

                    if not os.path.isfile(model_path):
                        model_path = os.path.join(self.model_base_dir, str(value))

                        if not os.path.isfile(model_path):
                            # Model is not available -> return error
                            print("--> Error: Model file not found.")
                            return

                    value = model_path

                args.append(arg_key)
                args.append(str(value))
                self.args_dict[arg_key] = value

        # Start the process using create_process
        self.create_process(name, self.target_server_app, *args)

    def create_process(self, name, executable_path, *args):
        if name in self.processes:
            print(f"--> Process with name '{name}' already exists.")
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
                print("--> Error: llama-server executable file not found.")
                return

            process = subprocess.Popen([executable_path, *args], preexec_fn=os.setsid)
            self.processes[name] = process

        print(f"--> Process '{name}' started with PID {process.pid}.")
        self.update_process_list_file()

    def stop_process(self, name):
        if name not in self.processes:
            print(f"--> No process found with name '{name}'.")
            return

        process = self.processes[name]
        if process.poll() is None:
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
                print(f"--> Process '{name}' with PID {process.pid} has been stopped.")
            except Exception as e:
                print(f"--> Failed to kill process group for '{name}': {e}")
        else:
            print(f"--> Process '{name}' is not running.")
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
            