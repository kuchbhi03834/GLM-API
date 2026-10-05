# GLM-API: Rock-Solid OpenAI-Compatible API for GLM (ChatGLM / Zhipu AI)

A production-grade, thread-safe, OpenAI-compatible proxy server for **GLM-4 / ChatGLM**, built with the exact same robust architecture as `Deepseek-API`.

Designed specifically for coding agents (**OpenCode**, **Cursor**, **Aider**, **Continue.dev**, **Cline**) and IDE tools.

Runs by default on **port `8001`**, allowing you to run both `Deepseek-API` (8000) and `GLM-API` (8001) simultaneously!

> 📖 **Looking for step-by-step setup instructions?** Check out the [Comprehensive Setup & IDE Integration Guide](SETUP_GUIDE.md).

---

## 🌟 Key Features

1. **Dual Mode Support**:
   - **Mode A: Free Web Session (`chatglm.cn`)**: Uses your personal account via browser session / Playwright. Supports auto-login, persistent cookie reuse, and automated session recovery.
   - **Mode B: Official Zhipu BigModel API**: Just provide your `ZHIPU_API_KEY` in `.env` if you have an official API key.

2. **Full OpenAI Compatibility**:
   - Both `/v1/chat/completions` and `/chat/completions` routes.
   - Both `/v1/models` and `/models` discovery endpoints.
   - Fallback `/v1/embeddings` stub so IDE tools never crash.
   - Diagnostics & health metrics at `/healthz` and `/`.

3. **Tool Calling & Multi-Turn Fidelity**:
   - Automatic OpenAI tool contract injection.
   - Multi-format parser extracting JSON markdown fences, bare JSON, XML tags, ReAct format (`Action:`/`Action Input:`), and function call syntax.
   - Automatic repair for truncated or streaming JSON.
   - Full multi-turn conversation preservation (replays past `tool_calls` and tool execution outputs).

4. **Thinking / Reasoning Tokens**:
   - Real-time streaming of thinking tokens as `delta.reasoning_content` for `glm-4-reasoner` (`glm-zero-preview`).

5. **Non-Blocking Concurrency Queue**:
   - Bounded task queue (`GLM_WORKERS=4`, `GLM_REQUEST_TIMEOUT=30s`) prevents server deadlocks under rapid IDE agent calls.
   - Returns polite HTTP 503 (`Retry-After: 2`) if queue capacity is momentarily exceeded.

6. **Browser Profile Reusability (Chrome, Opera, Brave, Edge)**:
   - Reuse your existing signed-in browser profile without logging in every time.

---

## 🚀 Quick Start

### 1. Requirements & Setup

```bash
cd "e:\Deepseek Api\GLM-API"
pip install -r requirements.txt
playwright install chromium
```

Copy the example environment file:
```bash
cp .env.example .env
```

### 2. Authentication (Choose One)

#### Option 1: Official Zhipu AI API Key (Easiest if you have one)
In your `.env`:
```env
ZHIPU_API_KEY=your_zhipu_api_key_here
```

#### Option 2: Free Web Account (`chatglm.cn`)
Run the interactive login script once:
```bash
python -m glm.auth
```
A browser window will open. Simply log into your `chatglm.cn` account (via phone number / WeChat / email). Once you reach the chat page, the session token is saved automatically to `.session_glm.json`.

*(Note: If you are already logged in on Opera, Chrome, or Edge, see [Using Other Browsers](#using-other-browsers) below).*

### 3. Start the Server

```bash
python app.py
```
Server starts on: `http://localhost:8001`

---

## 🛠️ Configuration & Models

### Available Model Aliases

| Model ID | Target Backend | Description |
| :--- | :--- | :--- |
| `glm-4-flash` | `glm-4-flash` | Ultra-fast free model (Default) |
| `glm-4` | `glm-4` | Flagship balanced model |
| `glm-4-plus` | `glm-4-plus` | Most capable flagship model |
| `glm-4-air` | `glm-4-air` | Lightweight high-speed model |
| `glm-4-reasoner` | `glm-zero-preview` | Reasoning model with deep thinking stream |
| `gpt-4o` | `glm-4-plus` | Alias for agent compatibility |
| `gpt-3.5-turbo` | `glm-4-flash` | Alias for agent compatibility |

---

## 🤖 Connecting to OpenCode / Cursor / Aider

### OpenCode Configuration
In OpenCode's settings or provider config:
- **Base URL**: `http://localhost:8001/v1`
- **API Key**: `not-needed` (or value of `API_KEY` if configured in `.env`)
- **Model**: `glm-4-flash` or `glm-4-plus` or `glm-4-reasoner`

### Python SDK Example
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[{"role": "user", "content": "Write a python function to sort a list."}],
)
print(response.choices[0].message.content)
```

---

## 🌐 Using Other Browsers (Opera, Chrome, Brave, Edge)

If your account is already logged into another browser like Opera or Chrome, you can point GLM-API directly to that browser's profile:

In `.env`:
```env
# Example for Opera:
GLM_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Roaming\Opera Software\Opera Stable"

# Example for Chrome:
GLM_PROFILE_DIR="C:\Users\<YourUsername>\AppData\Local\Google\Chrome\User Data"
```

Then run `python -m glm.auth` to capture the session seamlessly without re-typing credentials!

---

## 🧪 Running Tests

A complete automated unit and stress test suite is included:

```bash
py -3.13 -m unittest discover tests
```
All 24 test cases verify tool calling, multi-turn loops, streaming, error resilience, and queue concurrency.
