"""Comprehensive test suite for GLM-API server.

Tests:
- Tool calling prompt formatting and multi-turn thread handling
- Tool calling parsing across multiple formats (JSON, XML, ReAct, Call syntax)
- Streaming chunks with reasoning_content and content separation
- Concurrency task queue & timeouts
- Upstream error handling (rate limit, WAF block, session expiry)
- FastAPI endpoints compatibility with OpenAI clients
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import server.api as api
from glm.client import Chunk, Reply, UpstreamError
from glm.queue import GLMTaskQueue, GLMTimeoutError
from server.openai_format import (
    completion_response,
    extract_tool_calls,
    messages_to_prompt,
    stream_chunks,
    stream_chunks_with_tools,
    trim_at_role_boundary,
)
from server.schemas import ChatCompletionRequest, ChatMessage


class TestGLMToolCallingAndPrompt(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]

    def test_single_user_message_without_tools(self):
        msgs = [ChatMessage(role="user", content="Hello world")]
        prompt = messages_to_prompt(msgs)
        self.assertEqual(prompt, "Hello world")

    def test_messages_to_prompt_with_tools(self):
        msgs = [ChatMessage(role="user", content="Read test.txt")]
        prompt = messages_to_prompt(msgs, tools=self.tools)
        self.assertIn("# Tool calling", prompt)
        self.assertIn("read_file", prompt)
        self.assertIn("User: Read test.txt", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_multi_turn_with_assistant_tool_calls_and_results(self):
        msgs = [
            ChatMessage(role="user", content="Read test.txt"),
            ChatMessage(
                role="assistant",
                content="I will read the file.",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "test.txt"}',
                        },
                    }
                ],
            ),
            ChatMessage(
                role="tool",
                name="read_file",
                tool_call_id="call_123",
                content="file content: hello",
            ),
        ]
        prompt = messages_to_prompt(msgs, tools=self.tools)
        self.assertIn("Tool result (read_file): file content: hello", prompt)
        self.assertIn('"read_file"', prompt)
        self.assertIn('"path": "test.txt"', prompt)

    def test_tool_choice_required_and_pinned(self):
        msgs = [ChatMessage(role="user", content="Do something")]
        prompt_req = messages_to_prompt(msgs, tools=self.tools, tool_choice="required")
        self.assertIn("MUST call one of the tools above", prompt_req)

        prompt_pinned = messages_to_prompt(
            msgs,
            tools=self.tools,
            tool_choice={"type": "function", "function": {"name": "read_file"}},
        )
        self.assertIn("MUST call `read_file`", prompt_pinned)


class TestGLMToolCallExtraction(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {"type": "function", "function": {"name": "read_file", "parameters": {}}},
            {"type": "function", "function": {"name": "run_bash", "parameters": {}}},
        ]

    def test_extract_json_fence(self):
        raw = '```json\n{"tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}]}\n```'
        calls, leftover = extract_tool_calls(raw, self.tools)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "a.py"})
        self.assertEqual(leftover, "")

    def test_extract_bare_json(self):
        raw = '{"tool_calls": [{"name": "run_bash", "arguments": {"command": "ls"}}]}'
        calls, leftover = extract_tool_calls(raw, self.tools)
        self.assertIsNotNone(calls)
        self.assertEqual(calls[0]["function"]["name"], "run_bash")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls"})

    def test_extract_react_syntax(self):
        raw = 'Action: read_file\nAction Input: {"path": "main.py"}'
        calls, leftover = extract_tool_calls(raw, self.tools)
        self.assertIsNotNone(calls)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "main.py"})

    def test_extract_call_syntax(self):
        raw = 'I will execute: run_bash({"command": "pwd"})'
        calls, leftover = extract_tool_calls(raw, self.tools)
        self.assertIsNotNone(calls)
        self.assertEqual(calls[0]["function"]["name"], "run_bash")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "pwd"})

    def test_extract_xml_syntax(self):
        raw = '<tool_call name="read_file"><parameter name="path">config.json</parameter></tool_call>'
        calls, leftover = extract_tool_calls(raw, self.tools)
        self.assertIsNotNone(calls)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "config.json"})

    def test_normal_text_does_not_extract_calls(self):
        raw = "Hello! I am GLM-4. How can I assist you with your code today?"
        calls, leftover = extract_tool_calls(raw, self.tools)
        self.assertIsNone(calls)
        self.assertEqual(leftover, raw)


class TestGLMStreamingFormatters(unittest.TestCase):
    def test_stream_chunks_separates_thinking_and_content(self):
        chunks = [
            Chunk("Thinking step 1...", chunk_type="THINK"),
            Chunk("Thinking step 2...", chunk_type="THINK"),
            Chunk("Final answer here.", chunk_type="RESPONSE"),
        ]
        sse_lines = list(stream_chunks("glm-4-flash", chunks))
        content_deltas = []
        reasoning_deltas = []
        for line in sse_lines:
            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                payload = json.loads(line[len("data: "):])
                choices = payload.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "reasoning_content" in delta:
                        reasoning_deltas.append(delta["reasoning_content"])
                    if "content" in delta and delta["content"]:
                        content_deltas.append(delta["content"])

        self.assertEqual("".join(reasoning_deltas), "Thinking step 1...Thinking step 2...")
        self.assertEqual("".join(content_deltas), "Final answer here.")

    def test_stream_chunks_with_tools_emits_tool_calls(self):
        chunks = [
            Chunk("Thinking about files...", chunk_type="THINK"),
            Chunk('```json\n{"tool_calls": [{"name": "read_file", "arguments": {"path": "a.txt"}}]}\n```'),
        ]
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
        sse_lines = list(stream_chunks_with_tools("glm-4-flash", chunks, tools=tools))
        
        has_tool_call_delta = False
        has_reasoning = False
        for line in sse_lines:
            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                payload = json.loads(line[len("data: "):])
                choices = payload.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "reasoning_content" in delta:
                        has_reasoning = True
                    if "tool_calls" in delta:
                        has_tool_call_delta = True
                        self.assertEqual(delta["tool_calls"][0]["function"]["name"], "read_file")

        self.assertTrue(has_reasoning)
        self.assertTrue(has_tool_call_delta)


class TestGLMTaskQueue(unittest.TestCase):
    def test_queue_concurrency_and_acquire_release(self):
        queue = GLMTaskQueue(max_workers=2, timeout=1.0)
        self.assertTrue(queue.acquire(timeout=0.5))
        self.assertTrue(queue.acquire(timeout=0.5))
        self.assertEqual(queue.active_count, 2)
        # 3rd acquire should time out
        with self.assertRaises(GLMTimeoutError):
            queue.acquire(timeout=0.1)
        queue.release()
        self.assertEqual(queue.active_count, 1)
        queue.release()
        self.assertEqual(queue.active_count, 0)


class TestGLMUpstreamErrorHandling(unittest.TestCase):
    def test_rate_limit_detection(self):
        err1 = UpstreamError("Requests too frequent", status_code=429)
        self.assertTrue(err1.is_rate_limit)

        err2 = UpstreamError("API rate limit exceeded", finish_reason="rate_limit_reached")
        self.assertTrue(err2.is_rate_limit)

    def test_waf_detection(self):
        err = UpstreamError("Cloudflare verification or WAF check required", status_code=403)
        self.assertTrue(err.is_waf_block)


class TestGLMFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_healthz_endpoint(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "glm-openai-api")
        self.assertIn("models", data)
        self.assertIn("queue", data)

    def test_list_models_endpoints(self):
        for path in ("/models", "/v1/models"):
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["object"], "list")
            ids = [m["id"] for m in data["data"]]
            self.assertIn("glm-4-flash", ids)
            self.assertIn("glm-4-plus", ids)
            self.assertIn("glm-4-reasoner", ids)

    def test_embeddings_stub(self):
        resp = self.client.post("/v1/embeddings", json={"input": "Hello world"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 1)

    @patch("server.api.get_client")
    def test_chat_completions_non_streaming(self, mock_get_client):
        mock_glm_client = MagicMock()
        mock_glm_client.chat.return_value = Reply(
            text="Hello from GLM!",
            conversation_id="conv_123",
            reasoning_text="Thinking steps",
        )
        mock_get_client.return_value = mock_glm_client

        body = {
            "model": "glm-4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }
        resp = self.client.post("/v1/chat/completions", json=body)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["content"], "Hello from GLM!")
        self.assertEqual(data["choices"][0]["message"]["reasoning_content"], "Thinking steps")

    @patch("server.api.get_client")
    def test_chat_completions_streaming(self, mock_get_client):
        mock_glm_client = MagicMock()
        mock_glm_client.stream.return_value = iter([
            Chunk("Thinking...", chunk_type="THINK"),
            Chunk("Answer content", chunk_type="RESPONSE"),
        ])
        mock_get_client.return_value = mock_glm_client

        body = {
            "model": "glm-4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        resp = self.client.post("/v1/chat/completions", json=body)
        self.assertEqual(resp.status_code, 200)
        lines = resp.text.split("\n")
        self.assertTrue(any("reasoning_content" in l for l in lines))
        self.assertTrue(any("Answer content" in l for l in lines))


if __name__ == "__main__":
    unittest.main()
