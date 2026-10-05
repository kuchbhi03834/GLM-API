"""Pydantic models for the OpenAI-compatible request/response shapes for GLM-API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from .config import DEFAULT_MODEL


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict], None] = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    conversation_id: Optional[str] = None
    thinking: bool = False
    search: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
    stream_options: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None

    model_config = {"extra": "allow"}


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}
