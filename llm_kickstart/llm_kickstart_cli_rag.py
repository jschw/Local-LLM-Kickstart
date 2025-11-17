import argparse
import os, sys
import requests
import subprocess
import json
from llm_kickstart import LLMKickstart

def print_help():
    print("""Available commands:
        /start <name>        Start an endpoint with the given config name
        /stop <name>         Stop the endpoint with the given config name
        /restart <name>      Restart the endpoint with the given config name
        /stopall             Stop all running endpoints
        /status              Show status of all endpoints
        /listendpoints       List all available LLM endpoint configs
        /getconfig           Output the paths of app config and llm config files
        /editllm <Name of the llm config set> <name of the value> <New value>
        /editconf <Name of the key> <New value>
        /newllm <Name>         Create a new LLM config set
        /deletellm <Name>      Delete an LLM config set
        /renamellm <Old> <New> Rename an LLM config set
        /showllmconf <Name>   Show all parameters of the LLM config set <Name>
        /showappconf         Show all parameters in app_config.json
        /exportappconf <Path>    Export the loaded app_config.json to the given path
        /importappconf <Path>    Import a configuration file for app_config.json and refresh config
        /exportllmconf <Path>    Export the loaded llm_config.json to the given path
        /importllmconf <Path>    Import a configuration file for llm_config.json and refresh config
        /ragupdatefile <Path>    Read a PDF file into temp RAG system
        /ragupdatewebsite <URL> <Crawl ref depth> Read the content of a web page to temp RAG system
        /disablerag         Disables the temp RAG system
        /help                Show this help message
        /exit                Exit the CLI
        """)

def list_endpoints(llm_conf):
    try:
        if not llm_conf:
            print("No LLM endpoint configurations found.")
            return

        print("Available LLM endpoint configs:")

        for config in llm_conf:
            name = config.get("name", "Unnamed LLM")
            print(f"  - {name}")

    except Exception as e:
        print(f"Failed to load llm_config.json: {e}")

def main():
    parser = argparse.ArgumentParser(description="LLM Kickstart CLI - Manage inference endpoints")
    parser.add_argument('--start', metavar='NAME', type=str, help='Start the endpoint with the given config name on startup')
    args, _ = parser.parse_known_args()

    manager = LLMKickstart()

    # Get configs
    app_conf_path, app_conf = manager.get_app_config()
    llm_conf_path, llm_conf = manager.get_llm_config()

    if args.start:
        print(f"[Startup] Starting endpoint '{args.start}'...")
        manager.create_endpoint(args.start)

    # Start RAG proxy server
    rag_proxy_url = f"http://localhost:{app_conf['rag-proxy-serve-port']}"
    python_exec = sys.executable
    rag_server_process = subprocess.Popen([python_exec, "llm_kickstart_rag_server.py"])
    enable_rag_server = True

    print("LLM Kickstart CLI.\nType /help for commands. Type /exit to quit.\n")

    while True:
        try:
            user_input = input("llm-kickstart> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            rag_server_process.kill()
            break

        if not user_input:
            continue

        if not user_input.startswith("/"):
            print("Commands must start with a slash. Type /help for available commands.")
            continue

        tokens = user_input.split()
        command = tokens[0].lower()
        args = tokens[1:]

        if command == "/start":
            if len(args) != 1:
                print("Usage: /start <name>")
            else:
                manager.create_endpoint(args[0])
        
        elif command == "/stop":
            if len(args) != 1:
                print("Usage: /stop <name>")
            else:
                manager.stop_process(args[0])
        
        elif command == "/restart":
            if len(args) != 1:
                print("Usage: /restart <name>")
            else:
                manager.restart_process(args[0], None)
        
        elif command == "/stopall":
            manager.stop_all_processes()
        
        elif command == "/status":
            manager.list_processes()
        
        elif command == "/listendpoints":
            list_endpoints(llm_conf=llm_conf)

        elif command == "/getconfig":
            print(f"  -> App config path: {app_conf_path}")
            print(f"  -> LLM config path: {llm_conf_path}")
        
        elif command == "/help":
            print_help()
        
        elif command == "/exit":
            print("Exiting.")
            rag_server_process.kill()
            break
        
        elif command == "/editllm":
            if len(args) != 3:
                print("Usage: /editllm <Name of the llm config set> <name of the value> <New value>")
            else:
                config_set, key, value = args
                found = False
                for conf in llm_conf:
                    if conf.get("name") == config_set:
                        conf[key] = value
                        found = True
                        break
                if not found:
                    print(f"No LLM config set found with name '{config_set}'.")
                else:
                    try:
                        with open(llm_conf_path, "w") as f:
                            json.dump(llm_conf, f, indent=4)
                        manager.refresh_config()
                        print(f"Updated '{key}' in LLM config set '{config_set}' to '{value}'.")
                    except Exception as e:
                        print(f"Failed to update llm_config.json: {e}")

        elif command == "/editconf":
            if len(args) != 2:
                print("Usage: /editconf <Name of the key> <New value>")
            else:
                key, value = args
                if key not in app_conf:
                    print(f"Key '{key}' not found in app_config.json. Adding it.")
                app_conf[key] = value
                try:
                    with open(app_conf_path, "w") as f:
                        json.dump(app_conf, f, indent=4)
                    manager.refresh_config()
                    print(f"Updated '{key}' in app_config.json to '{value}'.")
                except Exception as e:
                    print(f"Failed to update app_config.json: {e}")

        elif command == "/newllm":
            if len(args) != 1:
                print("Usage: /newllm <Name>")
            else:
                new_name = args[0]
                # Check for duplicate
                if any(conf.get("name") == new_name for conf in llm_conf):
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
                    llm_conf.append(template)
                    try:
                        with open(llm_conf_path, "w") as f:
                            json.dump(llm_conf, f, indent=4)
                        manager.refresh_config()
                        print(f"Created new LLM config set '{new_name}'.")
                    except Exception as e:
                        print(f"Failed to create new LLM config set: {e}")

        elif command == "/deletellm":
            if len(args) != 1:
                print("Usage: /deletellm <Name>")
            else:
                del_name = args[0]
                found = False
                for i, conf in enumerate(llm_conf):
                    if conf.get("name") == del_name:
                        del llm_conf[i]
                        found = True
                        break
                if not found:
                    print(f"No LLM config set found with name '{del_name}'.")
                else:
                    try:
                        with open(llm_conf_path, "w") as f:
                            json.dump(llm_conf, f, indent=4)
                        manager.refresh_config()
                        print(f"Deleted LLM config set '{del_name}'.")
                    except Exception as e:
                        print(f"Failed to delete LLM config set: {e}")

        elif command == "/renamellm":
            if len(args) != 2:
                print("Usage: /renamellm <Old name> <New name>")
            else:
                old_name, new_name = args
                found = False
                for conf in llm_conf:
                    if conf.get("name") == old_name:
                        conf["name"] = new_name
                        found = True
                        break
                if not found:
                    print(f"No LLM config set found with name '{old_name}'.")
                else:
                    try:
                        with open(llm_conf_path, "w") as f:
                            json.dump(llm_conf, f, indent=4)
                        manager.refresh_config()
                        print(f"Renamed LLM config set from '{old_name}' to '{new_name}'.")
                    except Exception as e:
                        print(f"Failed to rename LLM config set: {e}")

        elif command == "/showllmconf":
            if len(args) != 1:
                print("Usage: /showllmconf <Name>")
            else:
                config_set = args[0]
                found = False
                for conf in llm_conf:
                    if conf.get("name") == config_set:
                        print(f"Parameters for LLM config set '{config_set}':")
                        for k, v in conf.items():
                            print(f"  {k}: {v}")
                        found = True
                        break
                if not found:
                    print(f"No LLM config set found with name '{config_set}'.")

        elif command == "/showappconf":
            print("Parameters in app_config.json:")
            for k, v in app_conf.items():
                print(f"  {k}: {v}")

        elif command == "/exportappconf":
            if len(args) != 1:
                print("Usage: /exportappconf <Path>")
            else:
                export_path = args[0]
                try:
                    with open(export_path, "w") as f:
                        json.dump(app_conf, f, indent=4)
                    print(f"-> Exported app_config.json to '{export_path}'.")
                except Exception as e:
                    print(f"-> Failed to export app_config.json: {e}")

        elif command == "/importappconf":
            if len(args) != 1:
                print("Usage: /importappconf <Path to json>")
            else:
                import_path = args[0]
                try:
                    with open(import_path, "r") as f:
                        imported_conf = json.load(f)
                    # Overwrite in-memory and file
                    app_conf.clear()
                    app_conf.update(imported_conf)
                    with open(app_conf_path, "w") as f:
                        json.dump(app_conf, f, indent=4)
                    manager.refresh_config()
                    print(f"-> Imported app_config.json from '{import_path}' and refreshed config.")
                except Exception as e:
                    print(f"-> Failed to import app_config.json: {e}")

        elif command == "/exportllmconf":
            if len(args) != 1:
                print("Usage: /exportllmconf <Path>")
            else:
                export_path = args[0]
                try:
                    with open(export_path, "w") as f:
                        json.dump(llm_conf, f, indent=4)
                    print(f"-> Exported llm_config.json to '{export_path}'.")
                except Exception as e:
                    print(f"-> Failed to export llm_config.json: {e}")

        elif command == "/importllmconf":
            if len(args) != 1:
                print("Usage: /importllmconf <Path to json>")
            else:
                import_path = args[0]
                try:
                    with open(import_path, "r") as f:
                        imported_conf = json.load(f)
                    # Overwrite in-memory and file
                    if isinstance(llm_conf, list):
                        llm_conf.clear()
                        llm_conf.extend(imported_conf)
                    else:
                        llm_conf = imported_conf
                    with open(llm_conf_path, "w") as f:
                        json.dump(llm_conf, f, indent=4)
                    manager.refresh_config()
                    print(f"-> Imported llm_config.json from '{import_path}' and refreshed config.")
                except Exception as e:
                    print(f"-> Failed to import llm_config.json: {e}")

        elif command == "/ragupdatefile":
            if len(args) != 1:
                print("Usage: /ragupdatefile <Path to PDF or txt file>")
            else:
                import_path = args[0]
                try:
                    if not os.path.exists(import_path):
                        print("Error: Document file not found.")
                        continue
                    
                    # Init temporary RAG system with file
                    payload = {"document_path": import_path}

                    response = requests.post(f"{rag_proxy_url}/v1/ragupdatepdf", json=payload)

                    print(f"-> RAG server status response: {response.json()["status"]}")
                       
                except Exception as e:
                    print(f"-> Failed to init RAG system: {e}")

        elif command == "/ragupdatewebsite":
            if len(args) != 2:
                print("Usage: /ragupdatewebsite <URL to website> <crawl ref depth>")
            else:
                url, crawl_depth = args
                try:
                    # Init temporary RAG system with web page content
                    pass
                       
                except Exception as e:
                    print(f"-> Failed to init RAG system: {e}")

        elif command == "/disablerag":
            # Disable a temporary RAG system
            try:
                response = requests.get(f"{rag_proxy_url}/v1/disablerag")

                print(f"-> RAG server status response: {response.json()["status"]}")
                    
            except Exception as e:
                print(f"-> Failed to turn off RAG system: {e}")

        else:
            print(f"Unknown command: {command}. Type /help for available commands.")

if __name__ == "__main__":
    main()
