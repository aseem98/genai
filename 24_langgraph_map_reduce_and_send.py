"""
=============================================================================
24_langgraph_map_reduce_and_send.py: LangGraph Send API & Dynamic Map-Reduce
=============================================================================
Senior GenAI Engineer Context:
How do you process a variable number of items concurrently in LangGraph?
For example, an agent breaks a query into 4 sub-questions, and you want
to run 4 specialized workers in parallel without hardcoding 4 static nodes.

The **LangGraph `Send` API**:
1. Map Phase: A generator node inspects state, identifies $N$ items, and emits
   $N$ dynamic `Send("worker_node", item)` commands.
2. Parallel Workers: LangGraph executes all $N$ workers concurrently.
3. Reduce Phase: A reducer node collects the parallel outputs and synthesizes
   the final report.

Run directly:
  python3 24_langgraph_map_reduce_and_send.py
=============================================================================
"""

import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient
JsonOutputParser = import_module("11_langchain_output_parsers_and_schemas").JsonOutputParser


@dataclass
class Send:
    """The LangGraph Send primitive for dynamic parallel routing."""
    node: str
    args: Dict[str, Any]


class MapReduceEngine:
    """Simulates LangGraph's dynamic Send-based map-reduce execution."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.json_parser = JsonOutputParser()

    def map_phase(self, topics: List[str]) -> List[Send]:
        """Emits a Send command for each topic."""
        print(f"{Colors.CYAN}▶ [Map Phase]: Emitting {len(topics)} dynamic Send commands...{Colors.RESET}")
        return [Send(node="analyze_stock_node", args={"ticker": t}) for t in topics]

    async def worker_node(self, send_cmd: Send) -> Dict[str, Any]:
        ticker = send_cmd.args["ticker"]
        print(f"  {Colors.YELLOW}⚡ [Worker Node ({ticker})]: Fetching fundamentals & valuation...{Colors.RESET}")

        prompt = (
            f"Give a rough fundamentals estimate for stock ticker {ticker} as JSON with keys "
            f"p_e_ratio (number) and recommendation (one of BUY, HOLD, SELL).\n{self.json_parser.get_format_instructions()}"
        )
        # `llm.chat` is a blocking network call — `asyncio.to_thread` is what
        # actually makes N of these run concurrently under one event loop,
        # instead of a bare `await` that would serialize them.
        raw = await asyncio.to_thread(
            self.llm.chat,
            [{"role": "user", "content": prompt}],
            simulated_response=json.dumps({"p_e_ratio": 28.4, "recommendation": "BUY"}),
        )
        try:
            parsed = self.json_parser.invoke(raw)
            return {"ticker": ticker, "p_e_ratio": parsed["p_e_ratio"], "recommendation": parsed["recommendation"]}
        except (ValueError, KeyError):
            return {"ticker": ticker, "p_e_ratio": None, "recommendation": f"UNPARSEABLE: {raw[:60]}"}

    @staticmethod
    def reduce_phase(results: List[Dict[str, Any]]) -> str:
        """Aggregates all parallel worker outputs into a single consolidated report."""
        print(f"\n{Colors.MAGENTA}▶ [Reduce Phase]: Aggregating parallel results from {len(results)} workers...{Colors.RESET}")
        summary = "Consolidated Portfolio Analysis:\n"
        for r in results:
            summary += f"  • {r['ticker']}: P/E = {r['p_e_ratio']} -> {r['recommendation']}\n"
        return summary


async def main():
    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("LANGGRAPH DYNAMIC MAP-REDUCE VIA `Send` PRIMITIVE")
    print(f"===================================================={Colors.RESET}\n")

    llm = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    portfolios = ["NVDA", "MSFT", "AAPL", "AMZN"]
    engine = MapReduceEngine(llm)

    # 1. Map: generate Send commands
    send_commands = engine.map_phase(portfolios)

    # 2. Concurrently execute all workers (each a real, independent LLM call)
    worker_tasks = [engine.worker_node(cmd) for cmd in send_commands]
    parallel_results = await asyncio.gather(*worker_tasks)

    # 3. Reduce: consolidate
    final_report = engine.reduce_phase(parallel_results)

    print(f"\n{Colors.BOLD}{Colors.GREEN}{final_report}{Colors.RESET}")


if __name__ == "__main__":
    asyncio.run(main())
