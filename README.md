# Chatshell

> **A conversational shell for local AI workflows**

`chatshell` is an open‑source, local‑first AI middleware that turns every chat with an LLM into a **connection layer between operating system and language models**.

With `chatshell`, conversations are not just messages - they are **commands** that can retrieve context, invoke tools, orchestrate agents, and automate workflows.

---

## Installation

```bash
pip install chatshell-python
```

## Configuration

On first run, `chatshell` will auto-generate configuration files in your user config directory (see [appdirs](https://pypi.org/project/appdirs/)):
- `chatshell_server_config.json` - Main server and RAG settings
- `llm_config.json` - LLM endpoint/model configurations

You can edit these files to customize document directories, model paths, and server options.

### Llama.cpp server binaries

#### MacOS
- Download prebuilt binaries for llama.cpp (see [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases))
- Allow prevented execution of the unsigned binaries:
  ```bash
  cd /Users/<current user>/chatshell/Llamacpp
  xattr -d com.apple.quarantine *
  ```

#### Linux
- Download prebuilt binaries or compile from source:
  ```bash
  git clone https://github.com/ggerganov/llama.cpp.git
  cd llama.cpp
  cmake -B build
  cmake --build build --config Release
  ```

## Usage

You can use `chatshell` as a CLI or as an OpenAI-compatible API server.

### CLI

Start the CLI:
```bash
./chatshell-server
```

You will see a prompt:
```
chatshell >
```
Type `/help` to see available commands.

### API Server

`chatshell` runs a FastAPI server that is OpenAI-compatible. By default, it listens on the port specified in your config (default: 4001).

You can send OpenAI API requests to:
```
http://localhost:4001/v1/chat/completions
```
and for the available model list:
```
http://localhost:4001/v1/models
```

### Example Usage

- Chat with a PDF:
  ```
  /chatwithfile mydoc.pdf
  ```
- Chat with a website:
  ```
  /chatwithwebsite https://example.com
  ```
- Summarize a document:
  ```
  /summarize mydoc.pdf
  ```
- Manage LLM endpoints:
  ```
  /listendpoints
  /startendpoint my-endpoint
  /stopendpoint my-endpoint
  ```


### Integrated commands and functions

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/chatwithfile <filename.pdf>` | Load a PDF or text file and chat with it |
| `/chatwithwebsite <URL>` | Load a website and chat with it |
| `/chatwithwebsite /deep <URL>` | Load a website, visit all sublinks, and chat with it |
| `/chatwithclipbrd` | Fetch content from clipboard and chat with the contents |
| `/summarize <filename.pdf or URL>` | Summarize a document or website and chat with the summary |
| `/summarize /clipboard` | Summarize the contents of the clipboard and chat with the summary |
| `/addclipboard` | Add the content of the clipboard to every message in the chat |
| `/forgetcontext` | Disable background injection of every kind of content |
| `/forgetall` | Disable RAG and all inserted contexts |
| `/forgetctx` | Disable inserted context only |
| `/forgetdoc` | Disable RAG (document/website context) only |
| `/updatemodels` | Update the LLM model catalog from GitHub |
| `/startendpoint <Endpoint config name>` | Start a specific LLM endpoint |
| `/restartendpoint <Endpoint config name>` | Restart a specific LLM endpoint |
| `/stopendpoint <Endpoint config name>` | Stop a specific LLM endpoint |
| `/stopallendpnts` | Stop all LLM inference endpoints |
| `/llmstatus` | Show the status of local LLM inference endpoints |
| `/setautostartendpoint <LLM endpoint name>` | Set a specific LLM endpoint for autostart |
| `/listendpoints` | List all available LLM endpoint configs |
| `/shellmode` | Activate shell mode for this chat (no LLM interaction) |
| `/exit` | Quit chatshell server |

