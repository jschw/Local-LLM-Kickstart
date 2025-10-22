# Local-LLM-Kickstart

A small tool for easily starting and stopping one or multiple local OpenAI-compatible LLM inference endpoints using `llama.cpp` server. Provides both a simple GUI and CLI for managing endpoints.

## Features & Options

### GUI Options

- **LLM Configuration Dropdown**: Select an LLM configuration (from `llm_config.json`) to use for starting an endpoint.
- **Start Endpoint**: Launches a new local LLM inference server with the selected configuration.
- **Stop Endpoint**: Stops the currently selected running endpoint.
- **Stop All Endpoints**: Stops all running endpoints at once.
- **Restart Endpoint**: Restarts the selected endpoint with its configuration.
- **Edit LLM Config**: Opens a configuration editor for modifying LLM settings.

### CLI Options

The CLI (via `llm_kickstart.py`) provides similar functionality programmatically:
- Start, stop, restart, and list endpoints.
- All configuration is read from `llm_config.json` and `app_config.json`.
- Example usage (Python):
  ```python
  from llm_kickstart import LLMKickstart
  manager = LLMKickstart()
  manager.create_endpoint("YourLLMName")
  manager.stop_process("YourLLMName")
  manager.stop_all_processes()
  ```

## Installation
Installation from git repo:

```
git clone https://github.com/jschw/Local-LLM-Kickstart
cd Local-LLM-Kickstart
pip install .
```

Install from pypy.org:

Commandline interface only:
```
pip install llm_kickstart
```
or
```
pip install llm_kickstart['cli']
```

With GUI:
```
pip install llm_kickstart['ui']
```

## Usage

### GUI

1. Run `llm_kickstart_ui.py` to launch the graphical interface.
2. Select an LLM configuration from the dropdown.
3. Use the buttons to start, stop, restart, or edit endpoints as needed.
4. Use "Refresh Endpoint List" to update the status display.
5. Only close the app using the "Close" button.

### CLI

- Import and use the `LLMKickstart` class in your own scripts, or extend the provided logic for custom automation.
- (Dedicated CLI application -> TODO)

## Configuration

- **llm_config.json**: Contains a list of LLM endpoint configurations (name, model path, and other options).
- **app_config.json**: Contains application-level settings, such as the path to the LLM server executable.

## Purpose

Local-LLM-Kickstart is designed for quick, user-friendly management of local LLM inference endpoints, making it easy to experiment with different models and settings via a GUI or scriptable CLI.
