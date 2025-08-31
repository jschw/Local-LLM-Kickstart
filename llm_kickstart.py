import multiprocessing
import subprocess
import os
import signal
import time
import json

# _run_executable is no longer needed, as we will use subprocess.Popen directly in create_process

class LLMKickstart:
    def __init__(self):
        self.target_server_app  = ""
        self.use_python_server_lib = False

        self.config             = None
        self.app_config         = None
        self.load_config()

        self.processes = {}
        self.process_list_file  = "process_list.json"
        self.output_cache       = ""  # Cache for all process output

        self.update_process_list_file()

    def load_config(self, llm_config_path="llm_config.json", app_config_path="app_config.json"):
        """
        Load and parse the llm_config.json file into structured variables.
        """
        try:
            with open(llm_config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Failed to load config file {llm_config_path}: {e}")
            self.config = None

        """
        Load and parse the app_config.json file into structured variables.
        """
        try:
            with open(app_config_path, "r") as f:
                self.app_config = json.load(f)
                self.target_server_app = self.app_config["llama-server-path"]
                self.use_python_server_lib = bool(self.app_config["use-llama-server-python"])

        except Exception as e:
            print(f"Failed to load config file {app_config_path}: {e}")
            self.app_config = None

    def create_endpoint(self, name):
        """
        Start a new process running ./llama_server with parameters from the config for the given LLM name.
        """
        if self.config is None:
            print("Configuration not loaded. Please call load_config() first.")
            return

        # Find the LLM config by name
        llm_config = None
        for conf in self.config:
            if conf.get("name") == name:
                llm_config = conf
                break

        if llm_config is None:
            print(f"No configuration found for LLM with name '{name}'.")
            return

        # Build command line arguments from the config
        args = []
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
                # if false, skip adding the flag
            else:
                args.append(arg_key)
                args.append(str(value))

        print(args)
        # Start the process using create_process
        self.create_process(name, self.target_server_app, *args)

    def create_process(self, name, executable_path, *args):
        if name in self.processes:
            print(f"Process with name '{name}' already exists.")
            return

        # Start the process in a new process group so we can kill all children
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
        for name, process in self.processes.items():
            status = "running" if process.poll() is None else "stopped"
            print(f"Process '{name}': PID {process.pid}, Status: {status}")
        self.update_process_list_file()

    def update_process_list_file(self):
        process_list = {
            name: {
                "pid": process.pid,
                "status": "running" if process.poll() is None else "stopped"
            }
            for name, process in self.processes.items()
        }
        with open(self.process_list_file, "w") as file:
            json.dump(process_list, file, indent=4)
            