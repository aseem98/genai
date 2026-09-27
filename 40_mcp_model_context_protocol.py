"""
=============================================================================
40_mcp_model_context_protocol.py: The Model Context Protocol (MCP)
=============================================================================
Senior GenAI Engineer Context:
MCP (Anthropic, 2024) standardizes how an LLM "Host" application talks to
external Tool/Resource "Servers" — it is the USB-C port for agents: instead
of writing bespoke tool-calling glue for every data source, any MCP-
compliant client can talk to any MCP-compliant server.

Architecture:
  Host (e.g. Claude Desktop / Claude Code)
    -> Client (1:1 connection per server)
      -> Server (exposes Tools, Resources, Prompts over JSON-RPC 2.0)

This module implements the protocol's message shapes and lifecycle exactly
as specified, over an in-process transport (a real deployment swaps this for
stdio or Streamable HTTP transports — the JSON-RPC payloads are identical):
  1. `initialize` handshake with capability negotiation.
  2. `tools/list` — server advertises its tools + JSON Schemas.
  3. `tools/call` — client invokes a tool by name with arguments.
  4. `resources/list` — server advertises addressable context (e.g. files).

Real equivalent (Python SDK):
  from mcp.server.fastmcp import FastMCP
  mcp = FastMCP("incident-tools")
  @mcp.tool()
  def get_runbook(service: str) -> str: ...

Run directly:
  python3 40_mcp_model_context_protocol.py
=============================================================================
"""

import json
from typing import Any, Callable, Dict, List
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors


# =====================================================================
# 1. JSON-RPC 2.0 ENVELOPE (the wire format MCP is built on)
# =====================================================================
def make_request(request_id: int, method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def make_result(request_id: int, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id: int, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


# =====================================================================
# 2. MCP SERVER: exposes Tools + Resources with capability negotiation
# =====================================================================
class MCPTool:
    def __init__(self, name: str, description: str, input_schema: dict, func: Callable):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.func = func

    def to_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "inputSchema": self.input_schema}


class MCPServer:
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.resources: List[dict] = []

    def tool(self, name: str, description: str, input_schema: dict):
        def decorator(func: Callable):
            self.tools[name] = MCPTool(name, description, input_schema, func)
            return func
        return decorator

    def add_resource(self, uri: str, description: str):
        self.resources.append({"uri": uri, "description": description})

    def handle(self, request: dict) -> dict:
        method, params, rid = request["method"], request.get("params", {}), request["id"]

        if method == "initialize":
            return make_result(rid, {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": self.name, "version": self.version},
                "capabilities": {"tools": {}, "resources": {"listChanged": True}},
            })

        if method == "tools/list":
            return make_result(rid, {"tools": [t.to_schema() for t in self.tools.values()]})

        if method == "tools/call":
            tool_name, args = params["name"], params.get("arguments", {})
            tool = self.tools.get(tool_name)
            if tool is None:
                return make_error(rid, -32601, f"Unknown tool: {tool_name}")
            try:
                output = tool.func(**args)
            except TypeError as e:
                return make_error(rid, -32602, f"Invalid arguments for {tool_name}: {e}")
            return make_result(rid, {"content": [{"type": "text", "text": str(output)}], "isError": False})

        if method == "resources/list":
            return make_result(rid, {"resources": self.resources})

        return make_error(rid, -32601, f"Method not found: {method}")


# =====================================================================
# 3. MCP CLIENT: discovers + invokes server capabilities on a Host's behalf
# =====================================================================
class MCPClient:
    def __init__(self, server: MCPServer):
        self.server = server  # stand-in for a stdio/HTTP transport
        self._next_id = 1
        self.server_capabilities: dict = {}

    def _call(self, method: str, params: dict) -> dict:
        request = make_request(self._next_id, method, params)
        self._next_id += 1
        response = self.server.handle(request)
        if "error" in response:
            raise RuntimeError(f"MCP error {response['error']['code']}: {response['error']['message']}")
        return response["result"]

    def initialize(self) -> dict:
        result = self._call("initialize", {
            "protocolVersion": "2025-06-18",
            "clientInfo": {"name": "agentic-host", "version": "1.0.0"},
            "capabilities": {},
        })
        self.server_capabilities = result["capabilities"]
        return result

    def list_tools(self) -> List[dict]:
        return self._call("tools/list", {})["tools"]

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        return result["content"][0]["text"]

    def list_resources(self) -> List[dict]:
        return self._call("resources/list", {})["resources"]


# =====================================================================
# 4. EXAMPLE MCP SERVER: incident-response tools
# =====================================================================
incident_tools = MCPServer(name="incident-tools-server")


@incident_tools.tool(
    name="get_runbook",
    description="Fetch the on-call runbook for a given service.",
    input_schema={"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
)
def get_runbook(service: str) -> str:
    return f"Runbook for '{service}': 1) Check dashboards 2) Roll back last deploy 3) Page on-call."


@incident_tools.tool(
    name="lookup_service_owner",
    description="Return the team that owns a given service.",
    input_schema={"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
)
def lookup_service_owner(service: str) -> str:
    owners = {"payments-api": "Team Ledger", "auth-api": "Team Identity"}
    return owners.get(service, "Unknown owner — escalate to platform on-call.")


incident_tools.add_resource(uri="runbooks://payments-api", description="Payments service runbook doc")


if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}=== Model Context Protocol (MCP): Host <-> Client <-> Server ==={Colors.RESET}\n")

    client = MCPClient(incident_tools)

    print(f"{Colors.YELLOW}1. initialize — capability negotiation handshake{Colors.RESET}")
    init_result = client.initialize()
    print(f"  {json.dumps(init_result, indent=2)}\n")

    print(f"{Colors.YELLOW}2. tools/list — server advertises its tool schemas{Colors.RESET}")
    for tool in client.list_tools():
        print(f"  • {tool['name']}: {tool['description']}")
    print()

    print(f"{Colors.YELLOW}3. tools/call — client invokes 'get_runbook'{Colors.RESET}")
    print(f"  -> {client.call_tool('get_runbook', {'service': 'payments-api'})}\n")

    print(f"{Colors.YELLOW}4. tools/call — client invokes 'lookup_service_owner'{Colors.RESET}")
    print(f"  -> {client.call_tool('lookup_service_owner', {'service': 'auth-api'})}\n")

    print(f"{Colors.YELLOW}5. resources/list — server advertises addressable context{Colors.RESET}")
    for resource in client.list_resources():
        print(f"  • {resource['uri']} — {resource['description']}")

    print(f"\n{Colors.GREEN}✔ Full MCP lifecycle (initialize -> discover -> invoke) verified.{Colors.RESET}")
    print(f"{Colors.DIM}Live equivalent: `pip install mcp` and swap MCPClient's transport for stdio_client(...).{Colors.RESET}")
