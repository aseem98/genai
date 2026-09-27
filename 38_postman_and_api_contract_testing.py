"""
=============================================================================
38_postman_and_api_contract_testing.py: Postman Collections & Contract Testing
=============================================================================
Senior GenAI Engineer Context:
Postman is the standard tool teams use to (a) hand-test an agent's HTTP API
during development and (b) run automated "collection" checks in CI via
Newman so a schema-breaking change never reaches production silently.

Two things make this useful for an agent service specifically:
  1. Generating a Postman Collection (v2.1 JSON schema) directly from your
     route definitions keeps the collection from drifting out of sync with
     the actual API — exactly what you'd otherwise get "for free" from
     FastAPI's OpenAPI export via a converter.
  2. A "collection runner" is just a sequence of (request, assertions)
     pairs executed against the API and checked for status code + shape —
     this module implements that runner in-process against the MiniAPI app
     from module 36, so it doubles as a CI contract-test suite.

Real equivalent:
  newman run agent_service.postman_collection.json --env-var base_url=...

Run directly:
  python3 38_postman_and_api_contract_testing.py
=============================================================================
"""

import json
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional
from importlib import import_module

fastapi_module = import_module("36_fastapi_service_layer_and_dependency_injection")
Colors = fastapi_module.Colors
app = fastapi_module.app


# =====================================================================
# 1. GENERATE A POSTMAN COLLECTION (v2.1 schema) FROM LIVE ROUTES
# =====================================================================
def route_to_postman_item(route) -> dict:
    body = None
    if route.body_model is not None and is_dataclass(route.body_model):
        sample = {f.name: (f.default if f.default is not None else "example") for f in fields(route.body_model)}
        body = {"mode": "raw", "raw": json.dumps(sample, default=str), "options": {"raw": {"language": "json"}}}

    return {
        "name": f"{route.method} {route.raw_path}",
        "request": {
            "method": route.method,
            "header": [{"key": "Content-Type", "value": "application/json"}],
            "url": {"raw": "{{base_url}}" + route.raw_path, "path": route.raw_path.split("/")},
            "body": body,
        },
    }


def build_postman_collection(api, collection_name: str) -> dict:
    return {
        "info": {
            "name": collection_name,
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "variable": [{"key": "base_url", "value": "http://localhost:8000"}],
        "item": [route_to_postman_item(route) for route in api.routes],
    }


# =====================================================================
# 2. COLLECTION RUNNER: executes each request in-process + asserts shape
#    (real equivalent: Newman's assertion-driven test scripts per request)
# =====================================================================
class ContractTest:
    def __init__(self, name: str, method: str, path: str, json_body: Optional[dict],
                 expect_status: int, expect_keys: Optional[List[str]] = None):
        self.name = name
        self.method = method
        self.path = path
        self.json_body = json_body
        self.expect_status = expect_status
        self.expect_keys = expect_keys or []


def run_collection(api, tests: List[ContractTest]) -> Dict[str, Any]:
    results = {"passed": 0, "failed": 0, "details": []}
    for test in tests:
        resp = api.request(test.method, test.path, test.json_body)
        errors = []
        if resp.status_code != test.expect_status:
            errors.append(f"expected status {test.expect_status}, got {resp.status_code}")
        for key in test.expect_keys:
            if not isinstance(resp.body, dict) or key not in resp.body:
                errors.append(f"missing expected key '{key}' in response body")

        passed = len(errors) == 0
        results["passed" if passed else "failed"] += 1
        results["details"].append({"test": test.name, "passed": passed, "errors": errors})
    return results


if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}=== Postman Collections & API Contract Testing ==={Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. Auto-generated Postman Collection (v2.1) from live routes{Colors.RESET}")
    collection = build_postman_collection(app, "Agentic Chat Service")
    print(json.dumps(collection, indent=2)[:900] + "\n  ... (truncated)\n")

    print(f"{Colors.YELLOW}2. Newman-style collection run (contract tests as CI gate){Colors.RESET}")
    suite = [
        ContractTest("Health check returns ok", "GET", "/health", None, 200, ["status"]),
        ContractTest("Chat with valid body succeeds", "POST", "/chat",
                     {"message": "ping", "agent_id": "qa-bot"}, 200, ["agent_id", "reply"]),
        ContractTest("Chat with empty message is rejected", "POST", "/chat", {"message": ""}, 422, ["detail"]),
        ContractTest("Unknown route returns 404", "GET", "/does-not-exist", None, 404, ["detail"]),
        # Intentionally-wrong expectation to show a failing assertion in the report:
        ContractTest("(demo) Health check wrongly expects 201", "GET", "/health", None, 201),
    ]
    report = run_collection(app, suite)

    for detail in report["details"]:
        icon = f"{Colors.GREEN}PASS{Colors.RESET}" if detail["passed"] else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  [{icon}] {detail['test']}")
        for err in detail["errors"]:
            print(f"         {Colors.RED}↳ {err}{Colors.RESET}")

    print(f"\n{Colors.BOLD}Summary: {Colors.GREEN}{report['passed']} passed{Colors.RESET}, "
          f"{Colors.RED}{report['failed']} failed{Colors.RESET} out of {len(suite)} contract tests.")
    print(f"{Colors.DIM}CI equivalent: fail the pipeline if `report['failed'] > 0`.{Colors.RESET}")
