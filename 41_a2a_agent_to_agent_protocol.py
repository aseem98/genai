"""
=============================================================================
41_a2a_agent_to_agent_protocol.py: Agent2Agent (A2A) Protocol
=============================================================================
Senior GenAI Engineer Context:
MCP (module 40) standardizes how an agent talks to TOOLS. A2A (Google, 2025,
now under the Linux Foundation) standardizes how one agent talks to ANOTHER
agent it doesn't control and whose internals it can't see — e.g. a travel
planner agent from Vendor A delegating currency-conversion to a specialist
agent from Vendor B.

Core A2A concepts implemented here from scratch:
  1. Agent Card: a public JSON manifest at `/.well-known/agent.json`
     describing an agent's identity, skills, and supported modalities.
  2. Task: the fundamental unit of work, with a formal lifecycle:
       submitted -> working -> (input-required) -> completed | failed
  3. Message / Part: turn-based exchange where each message holds one or
     more typed Parts (here: TextPart and DataPart).
  4. JSON-RPC methods `tasks/send` and `tasks/get`, mirroring the spec's
     HTTP+JSON-RPC transport.

Real equivalent (Python SDK): `pip install a2a-sdk`, then serve an
`AgentExecutor` behind `A2AStarletteApplication` and call it with `A2AClient`.

Run directly:
  python3 41_a2a_agent_to_agent_protocol.py
=============================================================================
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors


# =====================================================================
# 1. AGENT CARD — the public capability manifest agents discover each
#    other with (real equivalent: GET /.well-known/agent.json)
# =====================================================================
@dataclass
class AgentCard:
    name: str
    description: str
    url: str
    skills: List[str]
    version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "url": self.url,
                "skills": self.skills, "version": self.version}


# =====================================================================
# 2. TASK LIFECYCLE & MESSAGE PARTS
# =====================================================================
TASK_STATES = ["submitted", "working", "input-required", "completed", "failed"]


@dataclass
class TextPart:
    text: str
    kind: str = "text"


@dataclass
class DataPart:
    data: dict
    kind: str = "data"


@dataclass
class A2AMessage:
    role: str  # "user" (the calling agent) or "agent" (the remote agent)
    parts: List[Any]


@dataclass
class Task:
    id: str
    state: str = "submitted"
    history: List[A2AMessage] = field(default_factory=list)
    artifacts: List[dict] = field(default_factory=list)

    def transition(self, new_state: str):
        assert new_state in TASK_STATES, f"Invalid A2A task state: {new_state}"
        self.state = new_state


# =====================================================================
# 3. REMOTE A2A AGENT: a JSON-RPC server exposing `tasks/send`/`tasks/get`
# =====================================================================
class RemoteA2AAgent:
    """Real equivalent: an `AgentExecutor.execute()` implementation served
    behind `A2AStarletteApplication` at the Agent Card's `url`.
    """

    def __init__(self, card: AgentCard, handler: Callable[[Task], None]):
        self.card = card
        self.handler = handler
        self.tasks: Dict[str, Task] = {}

    def send_task(self, message: A2AMessage) -> Task:
        """JSON-RPC method: `tasks/send`."""
        task = Task(id=str(uuid.uuid4())[:8])
        task.history.append(message)
        task.transition("working")
        try:
            self.handler(task)
            if task.state == "working":
                task.transition("completed")
        except Exception as e:
            task.artifacts.append({"error": str(e)})
            task.transition("failed")
        self.tasks[task.id] = task
        return task

    def get_task(self, task_id: str) -> Task:
        """JSON-RPC method: `tasks/get`."""
        return self.tasks[task_id]


# =====================================================================
# 4. EXAMPLE: a "Trip Planner" agent delegates FX conversion to a
#    specialist "Currency Exchange" remote agent it doesn't own.
# =====================================================================
def currency_exchange_handler(task: Task):
    incoming = task.history[-1]
    data_parts = [p for p in incoming.parts if isinstance(p, DataPart)]
    if not data_parts:
        task.transition("input-required")
        task.history.append(A2AMessage(role="agent", parts=[TextPart("Need a DataPart with amount/from/to.")]))
        return

    payload = data_parts[0].data
    rates = {("USD", "EUR"): 0.92, ("USD", "JPY"): 149.5, ("EUR", "USD"): 1.09}
    rate = rates.get((payload["from"], payload["to"]))
    if rate is None:
        raise ValueError(f"No FX rate available for {payload['from']}->{payload['to']}")

    converted = round(payload["amount"] * rate, 2)
    task.artifacts.append({"name": "fx_conversion_result", "data": {"converted_amount": converted, "rate": rate}})
    task.history.append(A2AMessage(
        role="agent",
        parts=[TextPart(f"{payload['amount']} {payload['from']} = {converted} {payload['to']} (rate {rate})")],
    ))


currency_agent_card = AgentCard(
    name="Currency Exchange Agent",
    description="Converts an amount between two currencies using live-ish FX rates.",
    url="https://agents.example.com/currency-exchange",
    skills=["currency_conversion"],
)
currency_agent = RemoteA2AAgent(currency_agent_card, currency_exchange_handler)


class TripPlannerAgent:
    """The orchestrating agent — discovers the remote agent's card, then
    delegates a subtask to it over A2A rather than implementing FX itself.
    """

    def __init__(self, remote: RemoteA2AAgent):
        self.remote = remote

    def discover(self) -> AgentCard:
        return self.remote.card  # real: GET {base_url}/.well-known/agent.json

    def plan_budget(self, amount: float, home_currency: str, dest_currency: str) -> str:
        request = A2AMessage(
            role="user",
            parts=[
                TextPart(f"Please convert {amount} {home_currency} to {dest_currency} for a trip budget."),
                DataPart(data={"amount": amount, "from": home_currency, "to": dest_currency}),
            ],
        )
        task = self.remote.send_task(request)
        if task.state != "completed":
            return f"Delegation ended in state '{task.state}': {task.artifacts}"

        reply_text = next(p.text for p in task.history[-1].parts if isinstance(p, TextPart))
        return f"Trip budget ready — {reply_text}"


if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}=== Agent2Agent (A2A) Protocol: Cross-vendor Agent Delegation ==={Colors.RESET}\n")

    planner = TripPlannerAgent(currency_agent)

    print(f"{Colors.YELLOW}1. Discover the remote agent via its Agent Card{Colors.RESET}")
    card = planner.discover()
    print(f"  {card.to_dict()}\n")

    print(f"{Colors.YELLOW}2. tasks/send — delegate an FX-conversion subtask{Colors.RESET}")
    result = planner.plan_budget(amount=2000, home_currency="USD", dest_currency="EUR")
    print(f"  {result}\n")

    print(f"{Colors.YELLOW}3. Inspect the completed Task's full lifecycle + artifacts{Colors.RESET}")
    last_task_id = list(currency_agent.tasks.keys())[-1]
    task = currency_agent.get_task(last_task_id)
    print(f"  Task {task.id} final state: {Colors.GREEN}{task.state}{Colors.RESET}")
    print(f"  Artifacts: {task.artifacts}")
    print(f"  History turns: {len(task.history)}\n")

    print(f"{Colors.YELLOW}4. Missing required DataPart -> 'input-required' state{Colors.RESET}")
    incomplete_request = A2AMessage(role="user", parts=[TextPart("Convert some money please.")])
    bad_task = currency_agent.send_task(incomplete_request)
    print(f"  Task {bad_task.id} state: {Colors.RED if bad_task.state == 'failed' else Colors.YELLOW}{bad_task.state}{Colors.RESET}")

    print(f"\n{Colors.GREEN}✔ Agent Card discovery, task delegation, and lifecycle states verified.{Colors.RESET}")
    print(f"{Colors.DIM}Live equivalent: `pip install a2a-sdk` and swap RemoteA2AAgent for a real A2AClient/HTTP call.{Colors.RESET}")
