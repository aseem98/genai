"""
=============================================================================
25_multi_agent_collaboration_and_debate.py: Multi-Agent Debate & ADR Consensus
=============================================================================
Theory:
In enterprise decision making, complex trade-offs cannot be resolved by a
single persona.
In Multi-Agent Collaboration:
1. Product Manager (Elena) argues for user conversion and zero checkout friction.
2. Security Engineer (Marcus) argues for PCI-DSS compliance and client-side tokenization.
3. Principal Architect (Siddharth) balances sub-100ms latency and event-driven decoupling.
4. Two debate rounds run (opening positions, then rebuttals referencing the
   other personas), and the final Architecture Decision Record (ADR) is
   compiled directly from what was said in the rebuttal round.

Run directly:
  python3 25_multi_agent_collaboration_and_debate.py
=============================================================================
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient


@dataclass
class AgentPersona:
    name: str
    role: str
    color: str
    perspective: str
    rebuttal: str  # how this persona responds once it hears the others' opening positions


if __name__ == "__main__":
    client = LLMClient()

    pm = AgentPersona(
        "Elena", "Product Manager", Colors.BLUE,
        "Users abandon carts if checkout has > 2 steps.",
        "Given Marcus's PCI concern, I'll accept client-side tokenization as long as it stays a single step for the user."
    )
    sec = AgentPersona(
        "Marcus", "Security Architect", Colors.RED,
        "We cannot store raw card numbers. Must use PCI vault tokenization.",
        "Elena's 1-click flow works for me if the token exchange happens client-side before our servers ever see a PAN."
    )
    arch = AgentPersona(
        "Siddharth", "Principal Architect", Colors.YELLOW,
        "Use Kafka for async ledger and PostgreSQL read-replicas for <100ms latency.",
        "I can keep tokenization off the synchronous path entirely by publishing it onto the same Kafka ledger stream."
    )

    team = [pm, sec, arch]

    print(f"\n{Colors.BOLD}{Colors.HEADER}====================================================")
    print("COLLABORATIVE MULTI-AGENT ARCHITECTURE DEBATE")
    print(f"===================================================={Colors.RESET}\n")

    print(f"{Colors.BOLD}--- Round 1: Opening Positions ---{Colors.RESET}\n")
    for agent in team:
        # Routed through the LLM client (simulation-pinned here) so this
        # module's "agents" are actual calls, not just static strings.
        statement = client.chat(
            [{"role": "system", "content": f"You are {agent.name}, a {agent.role}."},
             {"role": "user", "content": "State your opening position on the payment checkout redesign."}],
            simulated_response=agent.perspective
        )
        print(f"{agent.color}{Colors.BOLD}[{agent.name} - {agent.role}]:{Colors.RESET}")
        print(f"  \"{statement}\"\n")

    print(f"{Colors.BOLD}--- Round 2: Rebuttals (each agent responds to the others) ---{Colors.RESET}\n")
    for agent in team:
        rebuttal = client.chat(
            [{"role": "system", "content": f"You are {agent.name}, a {agent.role}."},
             {"role": "user", "content": "Having heard the other two positions, respond and adjust your stance."}],
            simulated_response=agent.rebuttal
        )
        agent.rebuttal = rebuttal  # store what was actually said, for the facilitator to draw on
        print(f"{agent.color}{Colors.BOLD}[{agent.name} rebuttal]:{Colors.RESET}")
        print(f"  \"{rebuttal}\"\n")

    print(f"{Colors.BOLD}{Colors.GREEN}====================================================")
    print("★ CONSENSUS: ARCHITECTURE DECISION RECORD (ADR-001)")
    print(f"===================================================={Colors.RESET}\n")

    # Facilitator compiles the ADR directly from what was said in Round 2,
    # rather than a decision unrelated to the debate above.
    adr = "ADR-001: Scalable & Compliant Payment Processing Pipeline\n" + "\n".join(
        f"{i}. {agent.role} Decision ({agent.name}): {agent.rebuttal}"
        for i, agent in enumerate(team, 1)
    )
    print(adr)
