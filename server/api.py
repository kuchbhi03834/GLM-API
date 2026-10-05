"""
OpenAI-compatible FastAPI server for GLM (ChatGLM / Zhipu AI).

Supports both:
    - Free web session (chatglm.cn) via headless browser session or
    - Official Zhipu AI BigModel Open Platform API (ZHIPU_API_KEY).

Endpoints:
    POST /v1/chat/completions or /chat/completions
    GET  /v1/models or /models
    GET  /v1/models/{model} or /models/{model}
    POST /v1/embeddings or /embeddings
    GET  /healthz or /
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from glm.auth import DEFAULT_SESSION_FILE, LoginRequired, Session
from glm.client import GLMClient, UpstreamError
from glm.queue import GLMTimeoutError, get_glm_queue

from .config import (
    API_KEY,
    DEBUG_REQUESTS,
    GLM_REQUEST_TIMEOUT,
    GLM_WORKERS,
    MODEL_MAP,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    UPSTREAM_RETRY_AFTER,
    is_known_model,
    resolve_model_type,
    should_enable_thinking,
)
from .openai_format import (
    completion_response,
    extract_tool_calls,
    messages_to_prompt,
    stream_chunks,
    stream_chunks_with_tools,
    trim_at_role_boundary,
)
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest, EmbeddingRequest

load_dotenv()

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="GLM OpenAI-compatible API", version="0.1.0")

# Enable CORS for downstream coding agents (OpenCode, Cursor, Aider, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# Shared GLM client and task queue
_client: Optional[GLMClient] = None
_client_lock = threading.Lock()
_glm_queue = get_glm_queue(max_workers=GLM_WORKERS, timeout=GLM_REQUEST_TIMEOUT)


def get_client() -> GLMClient:
    """Build (once) the shared GLM client and session."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = GLMClient(
                    allow_interactive=SERVER_INTERACTIVE_LOGIN,
                    queue=_glm_queue,
                )
    return _client


def _error(
    message: str,
    status: int = 500,
    err_type: str = "server_error",
    headers: Optional[dict] = None,
):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": str(status)}},
        headers=headers or None,
    )


def _handle_upstream_error(exc: UpstreamError):
    """Translate upstream GLM errors into actionable status codes for clients."""
    if exc.is_rate_limit:
        return _error(
            f"GLM upstream rate limit reached: {exc}. Please slow down requests.",
            status=429,
            err_type="rate_limit_exceeded",
            headers={"Retry-After": str(UPSTREAM_RETRY_AFTER)},
        )
    if exc.is_waf_block:
        return _error(
            "GLM WAF / human verification check blocked this request. "
            "Please clear the human check or log in again by running 'python -m glm.auth'.",
            status=502,
            err_type="upstream_waf_block",
        )
    return _error(
        f"GLM reported an upstream error: {exc}",
        status=502,
        err_type="upstream_error",
    )


def _handle_queue_error(exc: Exception):
    """Graceful response when concurrency queue times out."""
    return _error(
        f"GLM request queue busy or timed out: {exc}. Please retry shortly.",
        status=503,
        err_type="queue_busy",
        headers={"Retry-After": "2"},
    )


def _verify_auth(authorization: Optional[str] = None) -> bool:
    """Optional API Key check if API_KEY is set in environment."""
    if not API_KEY:
        return True
    if not authorization:
        return False
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1] == API_KEY
    return authorization == API_KEY


class _Prefetched:
    """A stream with its first delta already pulled to surface errors early."""

    def __init__(self, stream, iterator, first):
        self._stream, self._iter, self._first = stream, iterator, first

    def __iter__(self):
        if self._first is not None:
            yield self._first
        yield from self._iter

    @property
    def conversation_id(self):
        return getattr(self._stream, "conversation_id", None)


@app.get("/")
@app.get("/healthz")
def healthz():
    """Detailed health check and diagnostics for GLM-API."""
    cached_session = Session.load(DEFAULT_SESSION_FILE)
    session_info = {
        "has_session": cached_session is not None,
        "is_api_key": bool(cached_session and cached_session.is_api_key),
        "age_seconds": round(cached_session.age, 1) if cached_session else None,
        "has_token": bool(cached_session and cached_session.token),
    }
    return {
        "status": "ok",
        "service": "glm-openai-api",
        "session": session_info,
        "queue": _glm_queue.stats(),
        "models": list(MODEL_MAP.keys()),
    }


@app.get("/models")
@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(None)):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": created,
                "owned_by": "zhipu",
                "permission": [],
                "root": name,
                "parent": None,
            }
            for name in MODEL_MAP
        ],
    }


@app.get("/models/{model}")
@app.get("/v1/models/{model}")
def retrieve_model(model: str, authorization: Optional[str] = Header(None)):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    if not is_known_model(model):
        return _error(f"The model `{model}` does not exist.", status=404, err_type="model_not_found")

    return {
        "id": model,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "zhipu",
        "permission": [],
        "root": model,
        "parent": None,
    }


@app.post("/embeddings")
@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest, authorization: Optional[str] = Header(None)):
    """Fallback stub for OpenAI embeddings so clients don't crash."""
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    inputs = [req.input] if isinstance(req.input, str) else req.input
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.0] * 8,
                "index": i,
            }
            for i in range(len(inputs))
        ],
        "model": req.model or "text-embedding-ada-002",
        "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
    }


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    if not _verify_auth(authorization):
        return _error("Incorrect API key provided.", status=401, err_type="invalid_api_key")

    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. Available models: {', '.join(MODEL_MAP)}",
            status=404,
            err_type="model_not_found",
        )

    model_type = None if req.conversation_id else resolve_model_type(req.model)
    thinking_enabled = should_enable_thinking(req.model, req.thinking)

    # Tool calling configuration
    tools = None if req.tool_choice == "none" else req.tools
    prompt = messages_to_prompt(req.messages, tools, req.tool_choice)

    if DEBUG_REQUESTS:
        tool_names = [(t.get("function") or t).get("name") for t in (req.tools or [])]
        log.warning(
            "request: model=%s stream=%s thinking=%s roles=%s tools=%s tool_choice=%s",
            req.model,
            req.stream,
            thinking_enabled,
            [m.role for m in req.messages],
            tool_names,
            req.tool_choice,
        )

    try:
        client = await run_in_threadpool(get_client)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:
        return _error(f"Failed to initialise GLM session: {e}", status=503)

    include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))

    if req.stream:
        try:
            raw = client.stream(
                prompt,
                conversation_id=req.conversation_id,
                model=model_type,
                thinking=thinking_enabled,
            )
            it = iter(raw)
            first = await run_in_threadpool(lambda: next(it, None))
        except UpstreamError as e:
            return _handle_upstream_error(e)
        except GLMTimeoutError as e:
            return _handle_queue_error(e)
        except LoginRequired as e:
            return _error(str(e), status=503, err_type="login_required")
        except Exception as e:
            return _error(f"GLM request failed: {e}")

        def gen():
            stream = _Prefetched(raw, it, first)
            if tools:
                yield from stream_chunks_with_tools(
                    req.model, stream, tools, include_usage=include_usage, prompt=prompt
                )
            else:
                yield from stream_chunks(
                    req.model, stream, include_usage=include_usage, prompt=prompt
                )

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        reply = await run_in_threadpool(
            client.chat,
            prompt,
            req.conversation_id,
            model_type,
            thinking_enabled,
        )
    except UpstreamError as e:
        return _handle_upstream_error(e)
    except GLMTimeoutError as e:
        return _handle_queue_error(e)
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except Exception as e:
        return _error(f"GLM request failed: {e}")

    calls, text = extract_tool_calls(reply.text, tools)
    if not tools:
        text = trim_at_role_boundary(text)

    if DEBUG_REQUESTS:
        log.warning(
            "reply: calls=%s text=%r reasoning=%r",
            [(c["function"]["name"], c["function"]["arguments"]) for c in calls or []],
            (text or "")[:200],
            (reply.reasoning_text or "")[:200],
        )

    return completion_response(
        req.model,
        text,
        prompt,
        reply.conversation_id,
        reasoning_content=reply.reasoning_text,
        tool_calls=calls,
    )
