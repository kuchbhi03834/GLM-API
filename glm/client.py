"""
Pure-HTTP GLM chat client.

Speaks Zhipu AI / ChatGLM backend or official Open Platform directly.
Supports both streaming and non-streaming completions, automated session recovery,
and reasoning / thinking chunk separation.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import httpx

from .auth import (
    DEFAULT_PROFILE_DIR,
    DEFAULT_SESSION_FILE,
    LoginRequired,
    Session,
    get_session,
    refresh_session,
)
from .queue import GLMTaskQueue, get_glm_queue

WEB_BASE = "https://chatglm.cn"
WEB_STREAM_PATH = "/chatglm/backend-api/assistant/stream"
OFFICIAL_BASE = "https://open.bigmodel.cn/api/paas/v4"


class UpstreamError(RuntimeError):
    """An error GLM reported inside stream or via HTTP."""

    def __init__(self, message: str, finish_reason: str = "", status_code: int = 502):
        super().__init__(message or "GLM reported an upstream error")
        self.finish_reason = finish_reason or ""
        self.status_code = status_code

    @property
    def is_rate_limit(self) -> bool:
        msg = str(self).lower()
        return (
            "rate_limit" in self.finish_reason.lower()
            or "too frequent" in msg
            or "rate limit" in msg
            or self.status_code == 429
        )

    @property
    def is_waf_block(self) -> bool:
        msg = str(self).lower()
        return (
            "waf" in msg
            or "cloudflare" in msg
            or "human" in msg
            or "verify" in msg
            or (self.status_code == 403 and "unauthorized" not in msg)
        )


class Chunk(str):
    """Stream token tagged with chunk_type ('RESPONSE' or 'THINK')."""

    def __new__(cls, value: str, chunk_type: str = "RESPONSE"):
        obj = str.__new__(cls, value)
        obj.chunk_type = chunk_type
        return obj


@dataclass
class Reply:
    text: str
    conversation_id: str
    reasoning_text: Optional[str] = None

    def __str__(self) -> str:
        return self.text


class GLMClient:
    def __init__(
        self,
        session: Optional[Session] = None,
        allow_interactive: bool = True,
        profile_dir: Optional[Path] = None,
        session_file: Optional[Path] = None,
        queue: Optional[GLMTaskQueue] = None,
    ):
        self.profile_dir = profile_dir or DEFAULT_PROFILE_DIR
        self.session_file = session_file or DEFAULT_SESSION_FILE
        self.session = session or get_session(
            profile_dir=self.profile_dir,
            session_file=self.session_file,
            allow_interactive=allow_interactive,
        )
        self.queue = queue or get_glm_queue()
        self._refresh_lock = threading.Lock()
        base_url = OFFICIAL_BASE if self.session.is_api_key else WEB_BASE
        self._http = httpx.Client(
            base_url=base_url,
            headers=self._base_headers(),
            cookies=self.session.cookies,
            timeout=httpx.Timeout(120.0, read=300.0),
        )

    def _base_headers(self) -> dict:
        auth_val = self.session.token if self.session.token.startswith("Bearer ") else f"Bearer {self.session.token}"
        headers = {
            "authorization": auth_val,
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": self.session.user_agent or "GLM-API/1.0",
        }
        if not self.session.is_api_key:
            headers.update({
                "origin": WEB_BASE,
                "referer": f"{WEB_BASE}/",
                "x-app-version": "1.0.0",
            })
        return headers

    def _recover_session(self) -> bool:
        """Thread-safe background headless recovery using cached profile."""
        with self._refresh_lock:
            cached = Session.load(self.session_file)
            if cached and cached.token != self.session.token:
                self.session = cached
                self._update_http_client()
                return True

            try:
                new_session = refresh_session(
                    profile_dir=self.profile_dir,
                    session_file=self.session_file,
                )
                if new_session:
                    self.session = new_session
                    self._update_http_client()
                    return True
            except Exception as e:
                print(f"[client] Automated GLM session recovery failed: {e}")

            return False

    def _update_http_client(self):
        base_url = OFFICIAL_BASE if self.session.is_api_key else WEB_BASE
        self._http.base_url = httpx.URL(base_url)
        self._http.headers.update(self._base_headers())
        self._http.cookies.update(self.session.cookies)

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
    ) -> "_GLMStream":
        return _GLMStream(self, prompt, conversation_id, model or "glm-4-flash", thinking)

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
    ) -> Reply:
        s = self.stream(prompt, conversation_id=conversation_id, model=model, thinking=thinking)
        reasoning_chunks = []
        content_chunks = []
        for chunk in s:
            if getattr(chunk, "chunk_type", "RESPONSE") == "THINK":
                reasoning_chunks.append(chunk)
            else:
                content_chunks.append(chunk)
        text = "".join(content_chunks)
        reasoning_text = "".join(reasoning_chunks) if reasoning_chunks else None
        return Reply(text=text, conversation_id=s.conversation_id, reasoning_text=reasoning_text)

    def close(self):
        self._http.close()


class _GLMStream:
    def __init__(self, client: GLMClient, prompt: str, conversation_id: Optional[str],
                 model: str, thinking: bool):
        self._client = client
        self._prompt = prompt
        self._conversation_id = conversation_id or str(uuid.uuid4())
        self._model = model
        self._thinking = thinking

    def __iter__(self) -> Iterator[Chunk]:
        self._client.queue.acquire()
        try:
            yield from self._stream_with_recovery()
        finally:
            self._client.queue.release()

    def _stream_with_recovery(self) -> Iterator[Chunk]:
        attempt = 0
        while attempt < 2:
            attempt += 1
            if self._client.session.is_api_key:
                # Official BigModel OpenAI-compatible endpoint
                body = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": self._prompt}],
                    "stream": True,
                }
                path = "/chat/completions"
            else:
                # Web Chat endpoint
                body = {
                    "assistant_id": "65940acff9477709f8759c20",
                    "conversation_id": self._conversation_id,
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": self._prompt}],
                        }
                    ],
                }
                path = WEB_STREAM_PATH

            req = self._client._http.build_request("POST", path, json=body)
            resp = self._client._http.send(req, stream=True)

            if resp.status_code == 401 and attempt == 1:
                resp.close()
                if self._client._recover_session():
                    continue
                raise LoginRequired("GLM session expired and background recovery failed.")
            if resp.status_code == 429:
                resp.close()
                raise UpstreamError("GLM rate limit reached", finish_reason="rate_limit_reached", status_code=429)
            if resp.status_code == 403:
                text = resp.read().decode("utf-8", errors="replace")
                resp.close()
                raise UpstreamError(f"Access forbidden (WAF / verification): {text[:200]}", finish_reason="waf_block", status_code=403)

            resp.raise_for_status()
            try:
                for chunk_type, text in _parse_glm_sse(resp.iter_lines(), is_official=self._client.session.is_api_key):
                    yield Chunk(text, chunk_type)
            finally:
                resp.close()
            break

    @property
    def conversation_id(self) -> str:
        return self._conversation_id


def _parse_glm_sse(lines, is_official: bool = False) -> Iterator[Tuple[str, str]]:
    """Parse Server-Sent Events from GLM official or web stream."""
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if is_official:
            choices = obj.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield "THINK", reasoning
                content = delta.get("content")
                if content:
                    yield "RESPONSE", content
        else:
            # Web format: parts[].content[].text or delta text
            if obj.get("status") == "error":
                raise UpstreamError(obj.get("message", "Web API error"))

            parts = obj.get("parts") or []
            for part in parts:
                p_type = part.get("type", "RESPONSE")
                chunk_type = "THINK" if "think" in p_type.lower() else "RESPONSE"
                for c in part.get("content", []):
                    txt = c.get("text")
                    if txt:
                        yield chunk_type, txt

            # Direct delta fallback
            if "delta" in obj and isinstance(obj["delta"], str):
                yield "RESPONSE", obj["delta"]
            elif "content" in obj and isinstance(obj["content"], str):
                yield "RESPONSE", obj["content"]
