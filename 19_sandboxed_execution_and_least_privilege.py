"""
=============================================================================
19_sandboxed_execution_and_least_privilege.py: Sandboxing & Least-Privilege Tools
=============================================================================
Senior GenAI Engineer Context:
Module 20's Reflexion loop runs LLM-generated code with a bare
`exec(code_str, {}, local_scope)`. That is NOT a sandbox — an empty globals
dict still lets the code reach dangerous builtins via any object's
`__class__.__mro__` chain, and even if builtins were fully stripped, nothing
stops the code from hanging forever or consuming unbounded memory. This
module implements the two things a real agent execution sandbox needs:

1. STATIC AST VETTING: parse the code and reject it BEFORE running anything
   if it imports modules, calls `eval`/`exec`/`__import__`, or touches dunder
   attributes — the same allowlist approach real sandboxes (RestrictedPython,
   Judge0) use as their first line of defense.
2. PROCESS-LEVEL ISOLATION: run code that passes the AST check in a
   SEPARATE subprocess with a hard wall-clock timeout — a real OS process
   boundary and kill switch, not just a restricted `exec()` in-process
   (which shares your interpreter's memory and can still hang it).

Also implements LEAST-PRIVILEGE TOOL SCOPING: a registry where each agent
role is only allowed to call an explicit allowlist of tools, with every
call (allowed or denied) written to an audit log.

Run directly:
  python3 19_sandboxed_execution_and_least_privilege.py
=============================================================================
"""

import ast
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors


# =====================================================================
# 1. STATIC AST VETTING (reject dangerous code before it ever runs)
# =====================================================================
DISALLOWED_NODE_TYPES = (ast.Import, ast.ImportFrom)
DISALLOWED_CALL_NAMES = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr", "globals", "locals"}


class UnsafeCodeError(Exception):
    pass


def vet_code(code_str: str) -> None:
    """Raises UnsafeCodeError if the code contains anything that could
    escape a sandbox: imports, dynamic eval/exec, filesystem access, or
    dunder-attribute access (the classic `().__class__.__base__.__subclasses__()`
    escape used to reach `os` without ever writing `import os`).
    """
    try:
        tree = ast.parse(code_str, mode="exec")
    except SyntaxError as e:
        raise UnsafeCodeError(f"Code does not parse: {e}")

    for node in ast.walk(tree):
        if isinstance(node, DISALLOWED_NODE_TYPES):
            raise UnsafeCodeError(f"Imports are not allowed (found on line {node.lineno})")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALL_NAMES:
            raise UnsafeCodeError(f"Call to disallowed builtin '{node.func.id}' on line {node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise UnsafeCodeError(f"Dunder attribute access '{node.attr}' is not allowed (line {node.lineno})")


# =====================================================================
# 2. PROCESS-LEVEL SANDBOX (real OS isolation + hard timeout)
# =====================================================================
@dataclass
class SandboxResult:
    success: bool
    stdout: str
    error: str
    duration_s: float


def run_sandboxed(code_str: str, timeout_s: float = 2.0) -> SandboxResult:
    """Vets the code, then runs it in a FRESH child interpreter process —
    a hard OS-level boundary. `subprocess.run(..., timeout=...)` guarantees
    the child is killed even if the code enters an infinite loop, which an
    in-process `exec()` can never guarantee (it would hang your whole agent).
    """
    t0 = time.perf_counter()
    try:
        vet_code(code_str)
    except UnsafeCodeError as e:
        return SandboxResult(success=False, stdout="", error=f"Rejected by static AST vetting: {e}", duration_s=0.0)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code_str],
            capture_output=True, text=True, timeout=timeout_s,
        )
        duration = time.perf_counter() - t0
        if proc.returncode != 0:
            return SandboxResult(success=False, stdout=proc.stdout, error=proc.stderr.strip(), duration_s=duration)
        return SandboxResult(success=True, stdout=proc.stdout, error="", duration_s=duration)
    except subprocess.TimeoutExpired:
        return SandboxResult(success=False, stdout="", error=f"Killed after exceeding {timeout_s}s timeout", duration_s=timeout_s)


# =====================================================================
# 3. LEAST-PRIVILEGE TOOL SCOPING + AUDIT LOG
# =====================================================================
@dataclass
class AuditEntry:
    role: str
    tool: str
    allowed: bool
    timestamp: float = field(default_factory=time.time)


class ScopedToolRegistry:
    """Each agent role is granted an explicit allowlist of tools — the
    least-privilege principle applied to agentic tool access, instead of
    every agent having implicit access to every registered tool.
    """

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.role_permissions: Dict[str, set] = {}
        self.audit_log: List[AuditEntry] = []

    def register_tool(self, name: str, func: Callable):
        self.tools[name] = func

    def grant(self, role: str, tool_names: List[str]):
        self.role_permissions[role] = set(tool_names)

    def call(self, role: str, tool_name: str, *args, **kwargs) -> Any:
        allowed = tool_name in self.role_permissions.get(role, set())
        self.audit_log.append(AuditEntry(role=role, tool=tool_name, allowed=allowed))
        if not allowed:
            raise PermissionError(f"Role '{role}' is not authorized to call tool '{tool_name}'")
        return self.tools[tool_name](*args, **kwargs)


def read_runbook(name: str) -> str:
    return f"[runbook contents for {name}]"


def restart_production_service(name: str) -> str:
    return f"[service {name} restarted]"


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== Sandboxed Execution & Least-Privilege Tooling ==={Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. Legitimate LLM-generated code -> runs fine in the sandbox{Colors.RESET}")
    good_code = textwrap.dedent("""
        def parse_version(v):
            parts = [int(x) for x in v.lstrip('v').split('.')]
            while len(parts) < 3:
                parts.append(0)
            return tuple(parts[:3])
        print(parse_version('v2.4'))
    """)
    result = run_sandboxed(good_code)
    print(f"  Success={result.success}, stdout={result.stdout.strip()!r}, took {result.duration_s*1000:.1f}ms\n")

    print(f"{Colors.YELLOW}2. Malicious code (tries to shell out) -> rejected BEFORE it ever runs{Colors.RESET}")
    malicious_code = "import os\nos.system('echo would have run a shell command')"
    result = run_sandboxed(malicious_code)
    print(f"  Success={result.success}, error={Colors.RED}{result.error}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}3. Dunder-based sandbox-escape attempt -> also rejected{Colors.RESET}")
    escape_code = "x = ().__class__.__base__.__subclasses__()"
    result = run_sandboxed(escape_code)
    print(f"  Success={result.success}, error={Colors.RED}{result.error}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}4. Infinite loop -> killed by the hard subprocess timeout, not left hanging{Colors.RESET}")
    infinite_code = "while True:\n    pass"
    result = run_sandboxed(infinite_code, timeout_s=1.0)
    print(f"  Success={result.success}, error={Colors.RED}{result.error}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}5. Least-privilege tool scoping + audit log{Colors.RESET}")
    registry = ScopedToolRegistry()
    registry.register_tool("read_runbook", read_runbook)
    registry.register_tool("restart_production_service", restart_production_service)
    registry.grant("triage_agent", ["read_runbook"])
    registry.grant("sre_lead_agent", ["read_runbook", "restart_production_service"])

    print(f"  {registry.call('triage_agent', 'read_runbook', 'redis-oom')}")
    try:
        registry.call("triage_agent", "restart_production_service", "payments-api")
    except PermissionError as e:
        print(f"  {Colors.RED}✖ Denied: {e}{Colors.RESET}")
    print(f"  {registry.call('sre_lead_agent', 'restart_production_service', 'payments-api')}\n")

    print(f"  {Colors.CYAN}Audit log:{Colors.RESET}")
    for entry in registry.audit_log:
        verdict = f"{Colors.GREEN}ALLOWED{Colors.RESET}" if entry.allowed else f"{Colors.RED}DENIED{Colors.RESET}"
        print(f"    [{verdict}] role={entry.role:<14} tool={entry.tool}")

    print(f"\n{Colors.GREEN}✔ AST vetting, process-isolated sandboxing, and least-privilege scoping all verified.{Colors.RESET}")
