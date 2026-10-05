"""Advanced tests for GLM-API: multi-turn agent loops, queue concurrency, and CORS."""

import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import server.api as api
from glm.auth import Session
from glm.client import Chunk, GLMClient, Reply, UpstreamError
from glm.queue import GLMTaskQueue
from server.openai_format import messages_to_prompt
from server.schemas import ChatCompletionRequest, ChatMessage


class TestGLMMultiTurnAgentLoop(unittest.TestCase):
    def test_full_opencode_multi_step_agent_interaction(self):
        """Simulate OpenCode driving a 3-step loop."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run bash command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            },
        ]

        messages = [
            ChatMessage(role="user", content="Fix the bug in src/main.py"),
            ChatMessage(
                role="assistant",
                content="Let me check git status first.",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"cmd": "git status"}'},
                }],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="call_1",
                name="bash",
                content="modified: src/main.py",
            ),
            ChatMessage(
                role="assistant",
                content="Now reading src/main.py.",
                tool_calls=[{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "src/main.py"}'},
                }],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="call_2",
                name="read_file",
                content="def main(): return 1/0",
            ),
            ChatMessage(role="user", content="Did you find the bug?"),
        ]

        prompt = messages_to_prompt(messages, tools=tools)

        self.assertIn("User: Fix the bug in src/main.py", prompt)
        self.assertIn("Assistant: Let me check git status first.", prompt)
        self.assertIn('"name": "bash"', prompt)
        self.assertIn("Tool result (bash): modified: src/main.py", prompt)
        self.assertIn('"name": "read_file"', prompt)
        self.assertIn("Tool result (read_file): def main(): return 1/0", prompt)
        self.assertIn("User: Did you find the bug?", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))


class TestGLMConcurrentTaskQueue(unittest.TestCase):
    def test_concurrent_threads_acquiring(self):
        queue = GLMTaskQueue(max_workers=3, timeout=5.0)
        results = []

        def worker():
            queue.acquire()
            time.sleep(0.05)
            results.append(True)
            queue.release()

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 6)
        self.assertEqual(queue.active_count, 0)


class TestGLMCORSAndRateLimitHeaders(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_cors_preflight(self):
        resp = self.client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(resp.headers.get("access-control-allow-origin"), ("*", "http://localhost:3000"))

    def test_rate_limit_headers_present(self):
        resp = self.client.get("/v1/models")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("x-ratelimit-limit", resp.headers)
        self.assertIn("x-ratelimit-remaining", resp.headers)
        self.assertIn("x-ratelimit-reset", resp.headers)


if __name__ == "__main__":
    unittest.main()
