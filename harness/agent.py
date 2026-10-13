import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent
SERVER = str(BASE / "tools" / "server.py")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b" 

SYSTEM = (
    "You answer questions using the tools provided. "
    "Use tools for any data lookup or arithmetic. "
    "When you have the answer, reply with a short final answer and no tool call."
)
FALLBACK = "I could not finish this task within safe limits."


@dataclass
class Limits:
    max_steps: int = 8
    max_context_tokens: int = 3500    # estimated prompt size before each model call
    max_total_tokens: int = 20000     # tokens used across the whole run
    model_timeout: float = 60         # seconds per model call
    tool_timeout: float = 20          # seconds per tool call
    run_timeout: float = 180          # seconds for the whole run
    max_tool_chars: int = 2000        # tool output is cut to this length
    max_repeats: int = 3              # identical tool calls allowed before stopping


@dataclass
class AgentResult:
    answer: str
    stop_reason: str   # final, max_steps, context_budget, token_budget,
                       # run_timeout, model_error, repeat_loop
    steps: int
    total_tokens: int
    seconds: float


def estimate_tokens(messages) -> int:
    """Rough estimate: about 4 characters per token."""
    return len(json.dumps(messages)) // 4


def to_openai_tools(mcp_tools):
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


async def run_agent(question: str, limits: Limits | None = None,
                    verbose: bool = True, llm=None) -> AgentResult:
    limits = limits or Limits()
    llm = llm or client
    start = time.time()
    total_tokens = 0

    def finish(answer: str, reason: str, steps: int) -> AgentResult:
        if verbose and reason != "final":
            print(f"[guardrail] stopped: {reason}")
        return AgentResult(answer, reason, steps, total_tokens,
                           round(time.time() - start, 2))

    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = to_openai_tools((await session.list_tools()).tools)

            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ]
            seen = {}   # counts identical tool calls

            for step in range(1, limits.max_steps + 1):
                # Pre-flight checks, before spending anything on a model call.
                if time.time() - start > limits.run_timeout:
                    return finish(FALLBACK, "run_timeout", step - 1)
                if estimate_tokens(messages) > limits.max_context_tokens:
                    return finish(FALLBACK, "context_budget", step - 1)
                if total_tokens > limits.max_total_tokens:
                    return finish(FALLBACK, "token_budget", step - 1)

                try:
                    resp = llm.chat.completions.create(
                        model=MODEL, messages=messages, tools=tools,
                        timeout=limits.model_timeout,
                    )
                except Exception as e:
                    if verbose:
                        print(f"[model error] {type(e).__name__}: {e}")
                    return finish(FALLBACK, "model_error", step - 1)

                if resp.usage:
                    total_tokens += resp.usage.total_tokens
                msg = resp.choices[0].message

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

                if not msg.tool_calls:
                    return finish(msg.content or "", "final", step)

                for tc in msg.tool_calls:
                    key = (tc.function.name, tc.function.arguments)
                    seen[key] = seen.get(key, 0) + 1
                    if seen[key] >= limits.max_repeats:
                        return finish(FALLBACK, "repeat_loop", step)

                    args = {}
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        result = await asyncio.wait_for(
                            session.call_tool(tc.function.name, args),
                            timeout=limits.tool_timeout,
                        )
                        text = result.content[0].text if result.content else ""
                    except asyncio.TimeoutError:
                        text = "Error: the tool timed out."
                    except Exception as e:
                        text = f"Error: {e}"

                    if len(text) > limits.max_tool_chars:
                        text = text[: limits.max_tool_chars] + "\n[output truncated]"

                    if verbose:
                        print(f"[step {step}] {tc.function.name}({args}) -> {text[:200]}")
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": text}
                    )

            return finish(FALLBACK, "max_steps", limits.max_steps)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "Which country has the most customers, and how many more is that "
        "than the second-ranked country?"
    )
    r = asyncio.run(run_agent(question))
    print(f"\nFINAL: {r.answer}")
    print(f"stop_reason={r.stop_reason} steps={r.steps} "
          f"tokens={r.total_tokens} seconds={r.seconds}")
