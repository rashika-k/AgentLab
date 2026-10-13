import asyncio
import json
from types import SimpleNamespace as NS

from agent import Limits, run_agent


class FakeLLM:
    """A scripted stand-in for the model.
    mode="final": answers immediately with no tool call.
    mode="tool":  always asks for the calculator, never finishes.
    mode="error": always fails.
    vary=True makes each calculator call different (avoids repeat detection)."""

    def __init__(self, mode="tool", vary=False, tokens=100):
        self.mode, self.vary, self.tokens, self.n = mode, vary, tokens, 0
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.n += 1
        if self.mode == "error":
            raise RuntimeError("fake model failure")
        usage = NS(total_tokens=self.tokens)
        if self.mode == "final":
            msg = NS(content="42", tool_calls=None)
            return NS(choices=[NS(message=msg)], usage=usage)
        expr = f"{self.n} + 1" if self.vary else "1 + 1"
        call = NS(
            id=f"call_{self.n}",
            function=NS(name="calculator",
                        arguments=json.dumps({"expression": expr})),
        )
        return NS(choices=[NS(message=NS(content="", tool_calls=[call]))],
                  usage=usage)


Q = "Use the calculator to compute 1 + 1."

CASES = [
    ("final",          FakeLLM("final"),                   Limits()),
    ("max_steps",      FakeLLM("tool", vary=True),         Limits(max_steps=3)),
    ("repeat_loop",    FakeLLM("tool"),                    Limits(max_repeats=3)),
    ("token_budget",   FakeLLM("tool", vary=True, tokens=5000),
                                                           Limits(max_total_tokens=8000)),
    ("context_budget", FakeLLM("tool"),                    Limits(max_context_tokens=50)),
    ("run_timeout",    FakeLLM("tool"),                    Limits(run_timeout=0)),
    ("model_error",    FakeLLM("error"),                   Limits()),
]


async def main():
    for expected, llm, limits in CASES:
        r = await run_agent(Q, limits, verbose=False, llm=llm)
        status = "PASS" if r.stop_reason == expected else "FAIL"
        print(f"{status}  expected={expected:15} got={r.stop_reason:15} "
              f"steps={r.steps} answer={r.answer[:40]!r}")


asyncio.run(main())