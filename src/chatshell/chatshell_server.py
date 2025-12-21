import argparse
import appdirs
from pathlib import Path
import json
from .llm_server import LocalLLMServer
from .chatshell_core import Chatshell

# Define variables
CONFIG_DIR = Path(appdirs.user_config_dir(appname='chatshell'))
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

chatshell_app_config_path   = CONFIG_DIR / 'chatshell_app_config.json'
chatshell_app_config        = None

# Module config
enable_llm_server       = False
enable_webconfig        = False

def init():
    load_config()

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
        /showllmserverconf      Show all parameters in llm_server_config.json
        /help                Show this help message
        /exit                Exit the CLI
        """)
    
def load_config():
    global enable_llm_server
    """
    Load and parse the kickstart_config.json file into structured variables.
    """
    try:
        if not chatshell_app_config_path.exists():
            # Create llm config file if not existing
            # Template content of the llm_server_config.json
            tmp_kickstart_config = {
                "enable-llm-server": "True",
                }

            with chatshell_app_config_path.open('w') as f:
                json.dump(tmp_kickstart_config, f, indent=4)

        with open(chatshell_app_config_path, "r") as f:
            kickstart_config    = json.load(f)
            enable_llm_server   = json.loads(str(kickstart_config["enable-llm-server"]).lower())

    except Exception as e:
        print(f"Failed to load config file {chatshell_app_config_path}: {e}")
        kickstart_config = None

def main_app():
    global enable_llm_server

    parser = argparse.ArgumentParser(description="Chatshell CLI - Manage inference endpoints")
    parser.add_argument('--termux', action='store_true')
    parser.add_argument('--start', metavar='NAME', type=str, help='Start the endpoint with the given config name on startup')
    args, _ = parser.parse_known_args()

    if args.termux:
        print("--> Termux special paths enabled.")

    # Start modules
    llm_server = LocalLLMServer(termux_paths=args.termux)

    # Start Chatshell proxy server
    chatshell_server = Chatshell(termux_paths=args.termux)
    chatshell_server.start()
    chatshell_proxy_url = f"http://localhost:{chatshell_server.get_chatshell_proxy_serve_port()}"

    # Get config
    kickstart_config = None

    # Autostart specified endpoint
    if args.start:
        print(f"--> Starting endpoint '{args.start}'...")
        llm_server.create_endpoint(args.start)

    print("----> LLM Kickstart CLI.<----\n\n")
    print_help()

    while True:
        try:
            user_input = input("chatshell > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            chatshell_server.stop()
            llm_server.stop_all_processes()
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
                llm_server.create_endpoint(args[0])
        
        elif command == "/stop":
            if len(args) != 1:
                print("Usage: /stop <name>")
            else:
                llm_server.stop_process(args[0])
        
        elif command == "/restart":
            if len(args) != 1:
                print("Usage: /restart <name>")
            else:
                llm_server.restart_process(args[0], None)
        
        elif command == "/stopall":
            llm_server.stop_all_processes()
        
        elif command == "/status":
            llm_server.list_processes()
        
        elif command == "/listendpoints":
            llm_server.listendpoints()

        elif command == "/getconfig":
            #print(f"  -> App config path: {app_conf_path}")
            print(f"  -> LLM server config path: {llm_server.get_llm_server_config_path()}")
            print(f"  -> LLM config path: {llm_server.get_llm_config_path()}")
        
        elif command == "/help":
            print_help()
        
        elif command == "/exit":
            print("Exiting.")
            chatshell_server.stop()
            llm_server.stop_all_processes()
            break
        
        elif command == "/editllm":
            if len(args) != 3:
                print("Usage: /editllm <Name of the llm config set> <name of the value> <New value>")
            else:
                config_set, key, value = args
                llm_server.edit_llm_conf(config_set, key, value)

        elif command == "/editconf":
            if len(args) != 2:
                print("Usage: /editconf <Name of the key> <New value>")
            else:
                key, value = args
                llm_server.edit_llm_server_conf(key, value)

        elif command == "/newllm":
            if len(args) != 1:
                print("Usage: /newllm <Name>")
            else:
                new_name = args[0]
                llm_server.create_new_llm_config(new_name)

        elif command == "/deletellm":
            if len(args) != 1:
                print("Usage: /deletellm <Name>")
            else:
                del_name = args[0]
                llm_server.delete_llm_config(del_name)

        elif command == "/renamellm":
            if len(args) != 2:
                print("Usage: /renamellm <Old name> <New name>")
            else:
                old_name, new_name = args
                llm_server.rename_llm_config(old_name, new_name)

        elif command == "/showllmconf":
            if len(args) != 1:
                print("Usage: /showllmconf <Name>")
            else:
                config_set = args[0]
                llm_server.show_llm_config(config_set)

        elif command == "/showllmserverconf":
            llm_server.show_llm_server_config()

        else:
            print(f"Unknown command: {command}. Type /help for available commands.")

def main():
    init()
    main_app()

# ----------------------------
# Run main application
# ----------------------------
if __name__ == "__main__":
    main()
