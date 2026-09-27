"""
=============================================================================
11_langchain_output_parsers_and_schemas.py: Output Parsers, Schemas & Repair
=============================================================================
Language models emit text; downstream code needs typed data. Output parsers
are the bridge, and in production they need three layers:

1. EXTRACTION: pull the payload out of whatever the model wrapped it in —
   markdown fences, chatty preambles, trailing sign-offs.
   (`CommaSeparatedListOutputParser`, `JsonOutputParser`)
2. SCHEMA VALIDATION: check the payload against a declared schema —
   required fields, types, safe coercion ("4" -> 4) — and fail with a
   precise error otherwise. `SchemaOutputParser` builds its format
   instructions AND its validator from one dataclass, the same idea as
   LangChain's PydanticOutputParser.
3. REPAIR: when parsing fails, don't crash — send the bad output plus the
   exact error back to the model and ask it to fix it (`OutputFixingParser`,
   the LangChain pattern of the same name), with a bounded number of tries.

Run directly:
  python3 11_langchain_output_parsers_and_schemas.py
=============================================================================
"""

import dataclasses
import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Type
from importlib import import_module

lcel_module = import_module("10_langchain_lcel_runnables_and_pipes")
Runnable = lcel_module.Runnable
Colors = lcel_module.Colors
LLMClient = import_module("01_unified_llm_client").LLMClient


class OutputParserException(ValueError):
    pass


# =====================================================================
# 1. EXTRACTION PARSERS
# =====================================================================
class CommaSeparatedListOutputParser(Runnable):
    def get_format_instructions(self) -> str:
        return "Respond with ONLY a comma-separated list, e.g. `foo, bar, baz` — no numbering, no extra text."

    def invoke(self, text: str) -> List[str]:
        text = text.strip().strip("`").split("\n")[-1] if "\n" in text.strip() else text.strip()
        return [item.strip().strip(".") for item in text.split(",") if item.strip()]


class JsonOutputParser(Runnable):
    """Extracts JSON even if surrounded by conversational filler or markdown fences."""

    def get_format_instructions(self) -> str:
        return "Return a valid JSON object matching the requested schema. Do not include markdown codeblocks or explanations."

    def invoke(self, text: str) -> Any:
        # Bound the capture by the fence markers (not by brace counting), so
        # nested JSON inside the fence isn't truncated at the first `}`/`]`.
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            candidates = [match.group(1)]
        else:
            obj_match = re.search(r"\{.*\}", text, re.DOTALL)
            arr_match = re.search(r"\[.*\]", text, re.DOTALL)
            candidates = sorted((m.group(0) for m in (obj_match, arr_match) if m), key=len) + [text]

        last_error: Optional[json.JSONDecodeError] = None
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                last_error = e
        raise OutputParserException(f"Failed to parse valid JSON from LLM output: {last_error}\nRaw Output: {text}")


# =====================================================================
# 2. SCHEMA VALIDATION (PydanticOutputParser pattern, via dataclasses)
# =====================================================================
_TYPE_NAMES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array"}


class SchemaOutputParser(Runnable):
    def __init__(self, schema: Type):
        assert dataclasses.is_dataclass(schema)
        self.schema = schema
        self.json_parser = JsonOutputParser()

    def get_format_instructions(self) -> str:
        fields = ", ".join(f'"{f.name}": <{_TYPE_NAMES.get(f.type, "value")}>' for f in dataclasses.fields(self.schema))
        return f"Return ONLY a JSON object of the form {{{fields}}} with no other text."

    @staticmethod
    def _coerce(value: Any, target: type, field_name: str) -> Any:
        if isinstance(value, target) and not (target is int and isinstance(value, bool)):
            return value
        try:
            if target is int and isinstance(value, (str, float)) and float(value) == int(float(value)):
                return int(float(value))
            if target is float and isinstance(value, (str, int)) and not isinstance(value, bool):
                return float(value)
            if target is bool and isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
        except ValueError:
            pass
        raise OutputParserException(f"Field '{field_name}' expected {_TYPE_NAMES.get(target, target)}, got {value!r}")

    def invoke(self, text: str):
        data = self.json_parser.invoke(text)
        if not isinstance(data, dict):
            raise OutputParserException(f"Expected a JSON object, got {type(data).__name__}")
        kwargs, missing = {}, []
        for f in dataclasses.fields(self.schema):
            if f.name not in data:
                if f.default is dataclasses.MISSING:
                    missing.append(f.name)
                continue
            kwargs[f.name] = self._coerce(data[f.name], f.type, f.name)
        if missing:
            raise OutputParserException(f"Missing required field(s): {missing}")
        return self.schema(**kwargs)


# =====================================================================
# 3. REPAIR: OutputFixingParser
# =====================================================================
class OutputFixingParser(Runnable):
    def __init__(self, parser: SchemaOutputParser, llm: LLMClient, max_fix_attempts: int = 2):
        self.parser, self.llm, self.max_fix_attempts = parser, llm, max_fix_attempts
        self.fix_attempts_used = 0

    def invoke(self, text: str, simulated_fix: Optional[str] = None):
        self.fix_attempts_used = 0
        try:
            return self.parser.invoke(text)
        except OutputParserException as first_error:
            error = first_error
        for _ in range(self.max_fix_attempts):
            self.fix_attempts_used += 1
            text = self.llm.chat([{"role": "user", "content": (
                f"The following output failed to parse.\n\nOUTPUT:\n{text}\n\nERROR:\n{error}\n\n"
                f"Rewrite it so it satisfies these instructions exactly:\n{self.parser.get_format_instructions()}"
            )}], simulated_response=simulated_fix, temperature=0)
            try:
                return self.parser.invoke(text)
            except OutputParserException as e:
                error = e
        raise error


@dataclass
class DeploymentSpec:
    service_name: str
    port: int
    replicas: int
    public: bool = False


if __name__ == "__main__":
    llm = LLMClient()
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 11: Output Parsers, Schemas & Repair ==={Colors.RESET}")
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. CommaSeparatedListOutputParser on a live answer{Colors.RESET}")
    csv_parser = CommaSeparatedListOutputParser()
    raw = llm.chat([{"role": "user", "content": f"Name 5 open-source observability tools. {csv_parser.get_format_instructions()}"}],
                   simulated_response="Prometheus, Grafana, Jaeger, OpenTelemetry, Loki", temperature=0)
    tools = csv_parser.invoke(raw)
    print(f"  raw:    {raw!r}")
    print(f"  parsed: {tools} (type={type(tools).__name__}, {len(tools)} items)\n")

    print(f"{Colors.YELLOW}2. JsonOutputParser robustness tests on known-tricky inputs{Colors.RESET}")
    json_parser = JsonOutputParser()
    tricky_inputs = {
        "chatty preamble + fence": 'Here is the spec:\n```json\n{"service_name": "AuthService", "port": 8080}\n```\nHope this helps!',
        "nested object in fence": '```json\n{"scaling": {"min": 2, "max": 8}}\n```',
        "bare top-level array": '[{"step": 1}, {"step": 2}]',
    }
    for label, text in tricky_inputs.items():
        print(f"  {label:<26} -> {json_parser.invoke(text)}")
    try:
        json_parser.invoke("no json here at all")
    except OutputParserException:
        print(f"  {'no JSON at all':<26} -> {Colors.RED}raises OutputParserException (no silent garbage){Colors.RESET}")
    print()

    print(f"{Colors.YELLOW}3. SchemaOutputParser: instructions + validation from ONE dataclass{Colors.RESET}")
    spec_parser = SchemaOutputParser(DeploymentSpec)
    print(f"  Generated instructions: {spec_parser.get_format_instructions()}")
    raw = llm.chat([{"role": "user", "content": "Extract the deployment spec from: 'Deploy billing-api on port 9090 "
                     f"with 4 replicas, internal only.'\n{spec_parser.get_format_instructions()}"}],
                   simulated_response='{"service_name": "billing-api", "port": "9090", "replicas": 4, "public": false}', temperature=0)
    spec = spec_parser.invoke(raw)
    print(f"  raw:    {raw.strip()!r}")
    print(f"  typed:  {spec}  (port is {type(spec.port).__name__} — coerced if the model sent a string)")
    for bad in ('{"service_name": "x", "port": 80}', '{"service_name": "x", "port": "eighty", "replicas": 2}'):
        try:
            spec_parser.invoke(bad)
        except OutputParserException as e:
            print(f"  {Colors.RED}rejected {bad}: {e}{Colors.RESET}")
    print()

    print(f"{Colors.YELLOW}4. OutputFixingParser repairing a REAL malformed answer{Colors.RESET}")
    prose = llm.chat([{"role": "user", "content": "In one short sentence, describe this deployment: billing-api, port 9090, 4 replicas, not public."}],
                     simulated_response="The billing-api service runs on port 9090 with 4 replicas and is not publicly exposed.")
    print(f"  Asked WITHOUT format instructions, the model answered in prose:\n    {prose.strip()!r}")
    fixer = OutputFixingParser(spec_parser, llm)
    fixed = fixer.invoke(prose, simulated_fix='{"service_name": "billing-api", "port": 9090, "replicas": 4, "public": false}')
    print(f"  {Colors.GREEN}Repaired into a typed object after {fixer.fix_attempts_used} fix call(s): {fixed}{Colors.RESET}")

    print(f"\n{Colors.GREEN}✔ Extraction, schema validation, and LLM-driven repair all exercised on live output.{Colors.RESET}")
