import argparse
import sys
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
  /help                Show this help message
  /exit                Exit the CLI
""")

def list_endpoints():
    try:
        with open("llm_config.json", "r") as f:
            configs = json.load(f)
        if not configs:
            print("No LLM endpoint configurations found.")
            return
        print("Available LLM endpoint configs:")
        for config in configs:
            name = config.get("name", "Unnamed LLM")
            print(f"  - {name}")
    except Exception as e:
        print(f"Failed to load llm_config.json: {e}")

def main():
    parser = argparse.ArgumentParser(description="LLM Kickstart CLI - Manage inference endpoints")
    parser.add_argument('--start', metavar='NAME', type=str, help='Start the endpoint with the given config name on startup')
    args = parser.parse_args()

    manager = LLMKickstart()

    if args.start:
        print(f"[Startup] Starting endpoint '{args.start}'...")
        manager.create_endpoint(args.start)

    print("LLM Kickstart CLI. Type /help for commands. Type /exit to quit.")

    while True:
        try:
            user_input = input("llm-kickstart> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
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
            list_endpoints()
        
        elif command == "/help":
            print_help()
        
        elif command == "/exit":
            print("Exiting.")
            break
        
        else:
            print(f"Unknown command: {command}. Type /help for available commands.")

if __name__ == "__main__":
    main()
