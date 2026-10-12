import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = str(Path(__file__).resolve().parent / "server.py")

CALLS = [
    ("sql_query", {"query": "SELECT Country, COUNT(*) FROM Customer GROUP BY Country ORDER BY 2 DESC LIMIT 3"}),
    ("sql_query", {"query": "DELETE FROM Customer"}),
    ("calculator", {"expression": "(12.5 + 7) * 3"}),
    ("calculator", {"expression": "__import__('os').system('ls')"}),
    ("search_docs", {"query": "lru_cache maxsize"}),
]


async def main():
    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])
            for name, args in CALLS:
                result = await session.call_tool(name, args)
                print(f"\n{name} {args}\n  -> {result.content[0].text}")


asyncio.run(main())