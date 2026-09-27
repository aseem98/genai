"""
=============================================================================
15_native_tool_calling_and_schemas.py: Native Tool Calling — the Full Protocol
=============================================================================
Module 14's ReAct agent parsed tool calls out of free text. Modern APIs
(OpenAI function calling, Anthropic tool use) make tool calls a first-class,
structured part of the protocol:

1. SCHEMAS: each tool is declared as a JSON Schema — generated here from the
   Python signature, type hints, and the docstring's "Args:" section.
2. THE MODEL DECIDES: the reply carries `tool_calls` (possibly several in
   one turn — PARALLEL tool calls) instead of text.
3. VALIDATE, THEN EXECUTE: arguments are checked against the schema (missing
   / unknown / wrong-type) BEFORE the function runs. Errors — validation or
   business-rule — go back to the model as structured tool results, so it
   can recover instead of the program crashing.
4. RESULTS GO BACK: each result is a `{"role": "tool", "tool_call_id": ...}`
   message; the model is called again and either calls more tools or
   answers. This loop is the core of every tool-using agent.

The tools operate on a real SQLite database, and the demo verifies the
database state afterwards — tool calls have real side effects.

Run directly:
  python3 15_native_tool_calling_and_schemas.py
=============================================================================
"""

import inspect
import json
import re
import sqlite3
from typing import Any, Callable, Dict, List, Optional, get_type_hints
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


# =====================================================================
# 1. TOOL REGISTRY: schema generation + argument validation
# =====================================================================
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: List[Dict[str, Any]] = []

    @staticmethod
    def _arg_docs(doc: str) -> Dict[str, str]:
        """Parses a Google-style 'Args:' docstring section into {param: description}."""
        match = re.search(r"Args:\s*\n(.*?)(\n\s*\n|$)", doc, re.DOTALL)
        return dict(re.findall(r"^\s*(\w+):\s*(.+)$", match.group(1), re.MULTILINE)) if match else {}

    def register(self, func: Callable) -> Callable:
        doc = inspect.getdoc(func) or ""
        hints, arg_docs = get_type_hints(func), self._arg_docs(doc)
        properties, required = {}, []
        for name, param in inspect.signature(func).parameters.items():
            properties[name] = {"type": _JSON_TYPES.get(hints.get(name, str), "string"),
                                "description": arg_docs.get(name, name)}
            if param.default is inspect.Parameter.empty:
                required.append(name)
        self.tools[func.__name__] = func
        self.schemas.append({"type": "function", "function": {
            "name": func.__name__, "description": doc.split("\n")[0],
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}})
        return func

    def _schema(self, name: str) -> Dict[str, Any]:
        return next(s["function"]["parameters"] for s in self.schemas if s["function"]["name"] == name)

    def validate(self, name: str, args: Dict[str, Any]) -> Optional[str]:
        if name not in self.tools:
            return f"unknown tool '{name}'"
        schema = self._schema(name)
        missing = [r for r in schema["required"] if r not in args]
        unknown = [a for a in args if a not in schema["properties"]]
        wrong = [a for a, v in args.items() if a in schema["properties"] and
                 schema["properties"][a]["type"] == "number" and not isinstance(v, (int, float))]
        problems = ([f"missing required: {missing}"] if missing else []) + ([f"unknown arguments: {unknown}"] if unknown else []) \
            + ([f"expected numbers for: {wrong}"] if wrong else [])
        return "; ".join(problems) or None

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        problem = self.validate(name, args)
        if problem:
            return {"status": "error", "error": f"invalid call: {problem}"}
        try:
            return {"status": "success", "result": self.tools[name](**args)}
        except (ValueError, LookupError) as e:          # business-rule / not-found errors go back to the model
            return {"status": "error", "error": str(e)}


# =====================================================================
# 2. REAL TOOLS OVER A REAL DATABASE
# =====================================================================
DB = sqlite3.connect(":memory:", check_same_thread=False)
DB.executescript("""
CREATE TABLE customers (id TEXT PRIMARY KEY, name TEXT, tier TEXT, balance_usd REAL);
CREATE TABLE tickets   (id TEXT PRIMARY KEY, customer_id TEXT, subject TEXT, status TEXT);
CREATE TABLE credits   (tx_id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, amount_usd REAL, reason TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP);
INSERT INTO customers VALUES ('CUST-101', 'Sarah Connor', 'Enterprise', 1200.00), ('CUST-202', 'Miles Dyson', 'Starter', 40.00);
INSERT INTO tickets   VALUES ('TICK-902', 'CUST-101', 'API returned 503s for 40 minutes (SLA breach)', 'open'),
                             ('TICK-903', 'CUST-101', 'Invoice PDF missing logo', 'closed');
""")
MAX_CREDIT_WITHOUT_APPROVAL = 100.0
registry = ToolRegistry()


@registry.register
def get_customer(customer_id: str) -> dict:
    """Fetches a customer's name, subscription tier and current account balance.

    Args:
        customer_id: The customer id, e.g. CUST-101.
    """
    row = DB.execute("SELECT id, name, tier, balance_usd FROM customers WHERE id = ?", (customer_id,)).fetchone()
    if row is None:
        raise LookupError(f"no customer with id {customer_id}")
    return dict(zip(("id", "name", "tier", "balance_usd"), row))


@registry.register
def list_open_tickets(customer_id: str) -> list:
    """Lists the customer's open support tickets.

    Args:
        customer_id: The customer id, e.g. CUST-101.
    """
    rows = DB.execute("SELECT id, subject FROM tickets WHERE customer_id = ? AND status = 'open'", (customer_id,)).fetchall()
    return [{"ticket_id": r[0], "subject": r[1]} for r in rows]


@registry.register
def apply_account_credit(customer_id: str, amount_usd: float, reason: str) -> dict:
    """Applies a courtesy credit to a customer's balance (max $100 without manager approval).

    Args:
        customer_id: The customer id, e.g. CUST-101.
        amount_usd: Credit amount in US dollars; must be positive.
        reason: Short justification recorded in the ledger.
    """
    if amount_usd <= 0:
        raise ValueError("amount_usd must be positive")
    if amount_usd > MAX_CREDIT_WITHOUT_APPROVAL:
        raise ValueError(f"credits above ${MAX_CREDIT_WITHOUT_APPROVAL:.0f} require manager approval; nothing was applied")
    get_customer(customer_id)  # raises LookupError if the customer doesn't exist
    with DB:  # one transaction: ledger row + balance update succeed or fail together
        tx_id = DB.execute("INSERT INTO credits (customer_id, amount_usd, reason) VALUES (?, ?, ?)",
                           (customer_id, amount_usd, reason)).lastrowid
        DB.execute("UPDATE customers SET balance_usd = balance_usd + ? WHERE id = ?", (amount_usd, customer_id))
    return {"tx_id": tx_id, "new_balance_usd": get_customer(customer_id)["balance_usd"]}


# =====================================================================
# 3. THE TOOL-CALLING LOOP
# =====================================================================
def run_tool_loop(llm: LLMClient, user_request: str, scripted_turns: List[str], max_rounds: int = 6) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a support agent. Use the tools to look things up and act; never invent data."},
        {"role": "user", "content": user_request},
    ]
    scripted = list(scripted_turns)
    for round_no in range(1, max_rounds + 1):
        reply = llm.chat(messages, tools=registry.schemas, simulated_response=scripted.pop(0) if scripted else None, temperature=0)
        try:
            tool_calls = json.loads(reply).get("tool_calls", []) if reply.strip().startswith("{") else []
        except json.JSONDecodeError:
            tool_calls = []
        if not tool_calls:
            return reply  # the model answered in text: loop ends

        print(f"  {Colors.CYAN}round {round_no}: model requested {len(tool_calls)} tool call(s){Colors.RESET}")
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        for call in tool_calls:
            fn = call["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}") if isinstance(fn.get("arguments"), str) else (fn.get("arguments") or {})
            except json.JSONDecodeError:
                args = {"__unparseable__": fn.get("arguments")}
            result = registry.execute(fn["name"], args)
            color = Colors.GREEN if result["status"] == "success" else Colors.RED
            print(f"    {Colors.YELLOW}⚡ {fn['name']}({args}){Colors.RESET} -> {color}{json.dumps(result)[:140]}{Colors.RESET}")
            messages.append({"role": "tool", "tool_call_id": call.get("id", fn["name"]), "content": json.dumps(result)})
    return "(stopped: tool-call round limit reached)"


def tool_call(call_id: str, name: str, **args) -> Dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 15: Native Tool Calling — the Full Protocol ==={Colors.RESET}")
    llm = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. Schema generated from signature + type hints + docstring{Colors.RESET}")
    print("  " + json.dumps(registry.schemas[2], indent=2).replace("\n", "\n  ") + "\n")

    print(f"{Colors.YELLOW}2. Validation before execution (bad calls never reach the database){Colors.RESET}")
    print(f"  missing arg : {registry.execute('apply_account_credit', {'customer_id': 'CUST-101'})}")
    print(f"  wrong type  : {registry.execute('apply_account_credit', {'customer_id': 'CUST-101', 'amount_usd': 'fifty', 'reason': 'x'})}")
    print(f"  unknown tool: {registry.execute('delete_customer', {'customer_id': 'CUST-101'})}\n")

    balance_before = get_customer("CUST-101")["balance_usd"]
    print(f"{Colors.YELLOW}3. Full loop: lookup (parallel calls) -> act -> final answer{Colors.RESET}")
    request = "Customer CUST-101 had an SLA breach. Check their open tickets, and if there's an SLA-related one, apply a $50 courtesy credit."
    print(f"  User: {request}")
    answer = run_tool_loop(llm, request, scripted_turns=[
        json.dumps({"tool_calls": [tool_call("c1", "get_customer", customer_id="CUST-101"),
                                   tool_call("c2", "list_open_tickets", customer_id="CUST-101")]}),
        json.dumps({"tool_calls": [tool_call("c3", "apply_account_credit", customer_id="CUST-101", amount_usd=50.0,
                                             reason="SLA breach TICK-902")]}),
        "I confirmed open ticket TICK-902 (503s for 40 minutes, an SLA breach) and applied a $50 courtesy credit to Sarah Connor's account.",
    ])
    print(f"  {Colors.GREEN}Agent:{Colors.RESET} {answer}\n")

    print(f"{Colors.YELLOW}4. A business rule rejects the action — the model explains instead of the program crashing{Colors.RESET}")
    request = "Apply a $250 credit to CUST-101 for the outage."
    print(f"  User: {request}")
    answer = run_tool_loop(llm, request, scripted_turns=[
        json.dumps({"tool_calls": [tool_call("c4", "apply_account_credit", customer_id="CUST-101", amount_usd=250.0, reason="outage")]}),
        "I couldn't apply that: credits above $100 require manager approval, so nothing was changed. I can apply up to $100 now or escalate.",
    ])
    print(f"  {Colors.GREEN}Agent:{Colors.RESET} {answer}\n")

    print(f"{Colors.YELLOW}5. Verify the real side effects in the database{Colors.RESET}")
    ledger = DB.execute("SELECT tx_id, customer_id, amount_usd, reason FROM credits").fetchall()
    balance_after = get_customer("CUST-101")["balance_usd"]
    print(f"  credits ledger: {ledger}")
    print(f"  CUST-101 balance: ${balance_before:.2f} -> ${balance_after:.2f}")
    applied = sum(r[2] for r in ledger)
    ok = abs((balance_after - balance_before) - applied) < 1e-9 and all(r[2] <= MAX_CREDIT_WITHOUT_APPROVAL for r in ledger)
    print(f"  {Colors.GREEN + '✔ balance change equals the ledger, and no credit exceeded the approval limit.' if ok else Colors.RED + '✖ ledger/balance mismatch'}{Colors.RESET}")
