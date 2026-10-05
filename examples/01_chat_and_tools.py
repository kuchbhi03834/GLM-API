"""
Example: Using GLM-API with OpenAI Python SDK for Chat, Thinking, and Tool Calling.

Prerequisites:
  1. Start the GLM server:
     python app.py
     (Runs on http://localhost:8001)

  2. Run this script:
     python examples/01_chat_and_tools.py
"""

from openai import OpenAI

# Connect to the local GLM-API server
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed",  # or your API_KEY from .env if configured
)

print("=== 1. Basic Chat (glm-4-flash) ===")
response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[
        {"role": "user", "content": "Hello! Introduce yourself briefly in one sentence."}
    ],
)
print("Response:", response.choices[0].message.content)
print()

print("=== 2. Streaming with Reasoning / Thinking ===")
stream = client.chat.completions.create(
    model="glm-4-reasoner",
    messages=[
        {"role": "user", "content": "Which is bigger: 9.9 or 9.11? Explain."}
    ],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta:
        # If model is thinking:
        if getattr(delta, "reasoning_content", None):
            print(f"[Think: {delta.reasoning_content}]", end="", flush=True)
        # Content response:
        if getattr(delta, "content", None):
            print(delta.content, end="", flush=True)
print("\n")

print("=== 3. Tool Calling Agent ===")
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }
]

tool_response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[
        {"role": "user", "content": "What is the weather like in Tokyo right now?"}
    ],
    tools=tools,
)

msg = tool_response.choices[0].message
if msg.tool_calls:
    print("Tool Call Triggered!")
    for tc in msg.tool_calls:
        print(f"Function: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")
else:
    print("Assistant Response:", msg.content)
