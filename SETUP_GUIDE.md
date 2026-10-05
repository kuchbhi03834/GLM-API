# 🚀 GLM-API: Complete Setup & Deployment Guide

This guide will walk you through setting up and running **GLM-API** on your machine (Windows, macOS, or Linux).

GLM-API turns the **GLM-4 / ChatGLM** models from Zhipu AI into a local, OpenAI-compatible proxy server running on port **`8001`**.

---

## 📋 Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Installation & Virtual Environment](#2-installation--virtual-environment)
3. [Authentication (Choose One Mode)](#3-authentication-choose-one-mode)
   - [Mode A: Free Web Account (chatglm.cn)](#mode-a-free-web-account-chatglmcn)
   - [Mode B: Reusing Your Browser Profile (Chrome, Opera, Brave, Edge)](#mode-b-reusing-your-browser-profile)
   - [Mode C: Official Zhipu AI API Key](#mode-c-official-zhipu-ai-api-key)
4. [Starting the Local Server](#4-starting-the-local-server)
5. [Connecting to AI Coding Agents & IDEs](#5-connecting-to-ai-coding-agents--ides)
   - [OpenCode](#opencode)
   - [Cursor IDE](#cursor-ide)
   - [Aider](#aider)
   - [Continue.dev](#continuedev)
   - [Python OpenAI SDK](#python-openai-sdk)
6. [Available Models & Parameters](#6-available-models--parameters)
7. [Environment Variables Reference](#7-environment-variables-reference)
8. [Troubleshooting & FAQs](#8-troubleshooting--faqs)

---

## 1. Prerequisites

- **Python 3.9 to 3.13** installed.
- One of the following:
  - A free account on [chatglm.cn](https://chatglm.cn), OR
  - An official API key from [open.bigmodel.cn](https://open.bigmodel.cn).
- An internet connection.

---

## 2. Installation & Virtual Environment

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/GLM-API.git
cd GLM-API
```

### Step 2: Create a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
*(If PowerShell gives a script permission error, run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`)*

**On Windows (Command Prompt - CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### Step 4: Create Environment File
```bash
# Windows:
copy .env.example .env

# macOS / Linux:
cp .env.example .env
```

---

## 3. Authentication (Choose One Mode)

### Mode A: Free Web Account (`chatglm.cn`)
If you want to use the free consumer chat account:

1. Run the interactive login tool:
   ```bash
   python -m glm.auth
   ```
2. A Chromium browser window opens. Log into your `chatglm.cn` account (using phone number, WeChat, or email).
3. Once you land on the chat page, the session token is automatically captured and saved to `.session_glm.json`.
4. Close the browser window.

---

### Mode B: Reusing Your Browser Profile (Opera, Chrome, Brave, Edge)
If you are already logged into `chatglm.cn` in your primary browser, you can reuse the existing cookies and session without re-entering credentials!

Edit your `.env` file and set `GLM_PROFILE_DIR`:

#### 🌐 Windows Paths:
- **Opera / Opera GX**:
  ```env
  GLM_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Roaming\Opera Software\Opera Stable"
  # For Opera GX:
  # GLM_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Roaming\Opera Software\Opera GX Stable"
  ```
- **Google Chrome**:
  ```env
  GLM_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Local\Google\Chrome\User Data"
  ```
- **Brave Browser**:
  ```env
  GLM_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Local\BraveSoftware\Brave-Browser\User Data"
  ```
- **Microsoft Edge**:
  ```env
  GLM_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Local\Microsoft\Edge\User Data"
  ```

#### 🍏 macOS Paths:
- **Google Chrome**:
  ```env
  GLM_PROFILE_DIR="/Users/<YourUsername>/Library/Application Support/Google/Chrome"
  ```
- **Opera**:
  ```env
  GLM_PROFILE_DIR="/Users/<YourUsername>/Library/Application Support/com.operasoftware.Opera"
  ```

#### 🐧 Linux Paths:
- **Google Chrome**:
  ```env
  GLM_PROFILE_DIR="/home/<YourUsername>/.config/google-chrome"
  ```

> **Tip**: Make sure your browser is closed before running `python -m glm.auth` so file locks are released.

---

### Mode C: Official Zhipu AI API Key
If you have an official API key from Zhipu BigModel Open Platform (`open.bigmodel.cn`), simply put it in `.env`:
```env
ZHIPU_API_KEY="your_actual_api_key_here"
```
No browser or Playwright login is required when `ZHIPU_API_KEY` is provided!

---

## 4. Starting the Local Server

Run the server:
```bash
python app.py
```

The server starts on **`http://localhost:8001`**.

Verify it by opening your browser or running:
```bash
curl http://localhost:8001/healthz
```
Expected output:
```json
{
  "status": "ok",
  "service": "glm-openai-api",
  "session": {"has_session": true, "has_token": true},
  "queue": {"max_workers": 4, "active_requests": 0, "available_slots": 4},
  "models": ["glm-4-flash", "glm-4", "glm-4-plus", "glm-4-reasoner", "gpt-4o"]
}
```

---

## 5. Connecting to AI Coding Agents & IDEs

### OpenCode
In OpenCode settings or provider configuration:
- **Provider**: Custom / OpenAI Compatible
- **Base URL**: `http://localhost:8001/v1` (or `http://localhost:8001`)
- **API Key**: `not-needed` (or any random string)
- **Model**: `glm-4-flash` or `glm-4-plus` or `glm-4-reasoner`

### Cursor IDE
1. Open Cursor **Settings** -> **Models**.
2. Under **OpenAI API Key**, toggle **Override OpenAI Base URL**.
3. Set Base URL: `http://localhost:8001/v1`
4. Enter any dummy key: `sk-glm-local`
5. Add model names: `glm-4-flash`, `glm-4-plus`, and `glm-4-reasoner`.

### Aider
Run Aider with GLM:
```bash
aider --openai-api-base http://localhost:8001/v1 --openai-api-key not-needed --model glm-4-flash
```
For reasoning mode:
```bash
aider --openai-api-base http://localhost:8001/v1 --openai-api-key not-needed --model glm-4-reasoner
```

### Continue.dev
In your `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "GLM-4 Flash (Fast)",
      "provider": "openai",
      "model": "glm-4-flash",
      "apiBase": "http://localhost:8001/v1",
      "apiKey": "not-needed"
    },
    {
      "title": "GLM-4 Reasoner",
      "provider": "openai",
      "model": "glm-4-reasoner",
      "apiBase": "http://localhost:8001/v1",
      "apiKey": "not-needed"
    }
  ]
}
```

### Python OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[{"role": "user", "content": "Write a clean quicksort in Python."}],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```

---

## 6. Available Models & Parameters

| Model Identifier | Backend Target | Description |
| :--- | :--- | :--- |
| `glm-4-flash` | `glm-4-flash` | Ultra-fast free model (Default) |
| `glm-4` | `glm-4` | Standard balanced flagship model |
| `glm-4-plus` | `glm-4-plus` | Most capable flagship model |
| `glm-4-air` | `glm-4-air` | High-speed lightweight model |
| `glm-4-reasoner` | `glm-zero-preview` | Deep reasoning model with thinking tokens stream |
| `gpt-4o` | `glm-4-plus` | Alias for agent compatibility |
| `gpt-3.5-turbo` | `glm-4-flash` | Alias for agent compatibility |

---

## 7. Environment Variables Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8001` | HTTP port for the GLM server |
| `HOST` | `127.0.0.1` | Network interface (`0.0.0.0` for LAN access) |
| `ZHIPU_API_KEY` | *(None)* | Official BigModel API key (if using official backend) |
| `API_KEY` | *(None)* | Optional custom bearer key to protect your local server |
| `RATE_LIMIT_PER_MINUTE` | `60` | Request rate limit per IP |
| `GLM_WORKERS` | `4` | Concurrency worker limit for incoming requests |
| `GLM_REQUEST_TIMEOUT` | `30.0` | Max seconds before queue acquisition times out |
| `GLM_PROFILE_DIR` | *(None)* | Path to custom browser profile (Chrome/Opera/etc.) |
| `DEBUG_REQUESTS` | `0` | Set `1` to log incoming prompt & tool calls |

---

## 8. Troubleshooting & FAQs

### Q1: Can I run DeepSeek-API and GLM-API at the same time?
**Answer**: Yes! DeepSeek-API runs on port **`8000`** and GLM-API runs on port **`8001`**. Both can run in parallel on your computer without interfering with each other.

### Q2: What if my web session expires?
**Answer**: GLM-API has automatic background token recovery. If a new login is required:
```bash
python -m glm.auth
```
Once re-authorized, your server resumes seamlessly.

### Q3: How to test tool calling?
Run the bundled example:
```bash
python examples/01_chat_and_tools.py
```
This tests basic chat, streaming reasoning chunks, and tool calling with OpenAI schemas.
