import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent
SERVER = str(BASE / "tools" / "server.py")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"   # change if you pulled a different model

SYSTEM = (
    "You answer questions using the tools provided. "
    "Use tools for any data lookup or arithmetic. "
    "When you have the answer, reply with a short final answer and no tool call."
)


def to_openai_tools(mcp_tools):
    """Convert MCP tool descriptions into the format the chat API expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


async def run_agent(question: str, max_steps: int = 8, verbose: bool = True) -> str:
    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = to_openai_tools((await session.list_tools()).tools)

            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ]

            for step in range(1, max_steps + 1):
                resp = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools
                )
                msg = resp.choices[0].message

                # Record the model's reply in the history.
                assistant = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    assistant["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(assistant)

                # No tool request means this is the final answer.
                if not msg.tool_calls:
                    return msg.content or ""

                # Otherwise run each requested tool and add its result.
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        result = await session.call_tool(tc.function.name, args)
                        text = result.content[0].text if result.content else ""
                    except Exception as e:
                        text = f"Error: {e}"
                    if verbose:
                        print(f"[step {step}] {tc.function.name}({args if 'args' in locals() else '?'}) -> {text[:200]}")
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": text}
                    )

            return "Stopped: reached max steps without a final answer."


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "Which country has the most customers, and how many more is that "
        "than the second-ranked country?"
    )
    print("\nFINAL:", asyncio.run(run_agent(question)))