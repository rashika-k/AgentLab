import ast
import operator
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agentlab")

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "chinook.sqlite"
DOCS_DIR = BASE / "data" / "docs"


@mcp.tool()
def sql_query(query: str) -> str:
    """Run a read-only SQL SELECT query on the Chinook music store database.
    Tables include Customer, Invoice, InvoiceLine, Track, Album, Artist.
    Returns up to 50 rows as text."""
    if not query.strip().lower().startswith("select"):
        return "Error: only SELECT queries are allowed."
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = con.execute(query)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchmany(50)
        return f"columns: {cols}\nrows: {rows}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        con.close()


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if (isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '(12.5 + 7) * 3'.
    Supports + - * / // % ** and parentheses. Numbers only, no variables."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def search_docs(query: str) -> str:
    """Search the local markdown docs for lines containing every word in the query.
    Returns up to 5 matches as 'file:line: text'."""
    terms = query.lower().split()
    if not terms:
        return "Error: empty query."
    hits = []
    for path in sorted(DOCS_DIR.glob("**/*.md")):
        for n, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if all(t in line.lower() for t in terms):
                hits.append(f"{path.name}:{n}: {line.strip()[:200]}")
                if len(hits) == 5:
                    return "\n".join(hits)
    return "\n".join(hits) if hits else "No matches."


if __name__ == "__main__":
    mcp.run()