from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

tools = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]

resp = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "What is 12.5 * 3 + 7?"}],
    tools=tools,
)
msg = resp.choices[0].message
print(msg.tool_calls or msg.content)