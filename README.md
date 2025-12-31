<div align="center">
  <img src="./assets/Chatshell_Logo.png" alt="Logo" width="200">
  <h1 align="center">Chatshell: Local open-source interaction layer for AI workflows</h1>
</div>

**Chatshell** is a free and open-source application that provides **local Large Language Models (LLMs)** combined with **advanced Retrieval-Augmented Generation (RAG)** and **operating system integrations** - all controlled through a **chat-based, Discord-like interface**.

It runs quietly in the background and acts as a **middleware layer between the user, the operating system and AI models**, turning chat conversations into powerful, reproducible AI workflows.

No cloud lock-in.  
No hidden APIs.  
No dependency on big tech platforms.

---

## ✨ Why Chatshell?

- **Workflow-Oriented** - Chat is not just conversation, it’s orchestration  
- **Discord-like Interaction** - Commands and conversation live side-by-side
- **User Interface Idependent** - Every OpenAI-compatible Chat UI can be used (e.g. Jan or OpenWebUI)
- **Composable AI** - Combine tools, context, and models in a single chat flow 
- **Advanced RAG** - Chat with documents, websites, clipboard content, and more
- **Open Source & Independent** - No vendor lock-in, no proprietary backends
- **Local & Private** - Run everything entirely on your machine  

Chatshell is designed to be an **AI assistant framework**, not just another chat UI.
AI should be a **tool**, not a service you depend on.

---

## 💡 What Is Chatshell?

Chatshell is:

- A **conversational shell** for AI-powered workflows  
- A **local LLM runtime manager**  
- A **RAG engine** for documents, websites, and live content  
- An **OpenAI-compatible API server**  
- A **text-based control interface** for automation and analysis  

Chatshell is **not**:

- A hosted SaaS
- A closed ecosystem
- A single-purpose chatbot

---

## 🗨️ Discord-Like Chat Experience

Chatshell combines **natural conversation** with **command-based automation**:

You can just chat naturally and insert commands like:

```text
/summarize cat_manual.pdf
/chatwithwebsite https://allaboutcats.com
What are signs of a happy cat?
/forgetcontext
````

Commands are embedded directly into the chat flow, enabling:

* Context injection and removal
* Tool invocation
* Model and endpoint control
* Document and website analysis
* Automation without leaving the conversation

---

## 🚀 Features

### Local AI & Model Control

* Run local LLMs via llama.cpp
* Manage multiple inference endpoints
* Start, stop, restart models on demand
* Auto-start preferred model

### Advanced RAG

* Chat with PDFs and text files
* Chat with websites (shallow or deep crawl)
* Summarize documents or URLs
* Inject clipboard content into conversations

### Middleware Capabilities

* Acts as a bridge between LLM and OS
* Shell-like interaction
* OpenAI-compatible API for external tools

---

## 📦 Installation

```bash
pip install chatshell-python
```

---

## ⚙️ Configuration

On first run, Chatshell automatically creates configuration files in your user config directory (see [`appdirs`](https://pypi.org/project/appdirs/)):

* `chatshell_server_config.json` - Server, RAG, and runtime settings
* `llm_config.json` - LLM endpoints and model configurations

Edit these files to configure:

* Model paths
* Document directories
* Ports and server behavior
* Default endpoints

---

### 🦙 llama.cpp Binaries

Chatshell uses **llama.cpp** for local inference.
You can download the binaries from https://github.com/ggml-org/llama.cpp/releases and extract them to /home/user/chatshell/Llamacpp or use the llama.cpp python server bindings.

>**Note:** The python bindings are outdated at the moment and do not support Huggingface parameters. If you want to download models automatically from Huggingface, you have to use the current binaries.


### macOS

1. Download prebuilt binaries from the llama.cpp releases
2. Allow execution of unsigned binaries:

   ```
   cd /Users/<current user>/chatshell/Llamacpp
   xattr -d com.apple.quarantine *
   ```

### Linux

Compile from source:

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

>**Note:** This should also work in Termux on Android if you have installed all requirements. You can use the provided script termux_compile_llamacpp.sh.
---

## 🖥️ Usage

### CLI Mode

Start the interactive shell:

```
./chatshell-server
```

You’ll see:

```
chatshell >
```

Chatshell runs an **OpenAI-compatible FastAPI server**.
Configure your preferred Chat frontend with the default chatshell API endpoint:

```
http://localhost:4001/v1/chat/completions
```

You're ready now!

---

## 📚 Available Commands

| Command                            | Description                                 |
| ---------------------------------- | ------------------------------------------- |
| `/help`                            | Show this help message                      |
| `/chatwithfile <filename.pdf>`     | Load a PDF or text file and chat with it    |
| `/chatwithwebsite <URL>`           | Load a website and chat with it             |
| `/chatwithwebsite /deep <URL>`     | Load a website and all sublinks, then chat  |
| `/chatwithclipbrd`                 | Fetch clipboard content and chat with it    |
| `/summarize <filename.pdf or URL>` | Summarize a document or website             |
| `/summarize /clipboard`            | Summarize clipboard contents                |
| `/addclipboard`                    | Inject clipboard content into every message |
| `/forgetcontext`                   | Disable all background context injection    |
| `/forgetall`                       | Disable RAG and all inserted contexts       |
| `/forgetctx`                       | Disable inserted context only               |
| `/forgetdoc`                       | Disable document/website RAG only           |
| `/updatemodels`                    | Update model catalog from GitHub            |
| `/startendpoint <name>`            | Start a specific LLM endpoint               |
| `/restartendpoint <name>`          | Restart an LLM endpoint                     |
| `/stopendpoint <name>`             | Stop an LLM endpoint                        |
| `/stopallendpnts`                  | Stop all LLM endpoints                      |
| `/llmstatus`                       | Show endpoint status                        |
| `/setautostartendpoint <name>`     | Set endpoint for autostart                  |
| `/listendpoints`                   | List all endpoint configs                   |
| `/shellmode`                       | Enter shell-only mode (no LLM)              |
| `/exit`                            | Quit Chatshell                              |


---

## 📜 License

This project is released under an open-source license.
See `LICENSE` for details.

---

## 💡 Contributing

Contributions, ideas, and feedback are welcome.
Chatshell is meant to evolve as a community-driven AI workflow platform.

---
