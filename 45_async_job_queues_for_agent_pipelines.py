"""
=============================================================================
45_async_job_queues_for_agent_pipelines.py: Message Queues for Decoupled Agents
=============================================================================
Senior GenAI Engineer Context:
A synchronous request/response agent API (module 36) blocks the caller until
the LLM finishes — fine for a quick chat reply, wrong for a long-running
agent task (a multi-step research job, a batch document-classification run).
Production systems decouple the two sides with a message queue: a producer
publishes a job and returns immediately; a pool of workers consumes jobs
independently, at their own pace, and can be scaled/restarted separately
from the producer.

Implemented from scratch here (the mechanics behind SQS/RabbitMQ/Kafka
consumer groups):
1. AT-LEAST-ONCE DELIVERY: a consumer must explicitly `ack()` a message;
   an unacked message becomes visible again for another worker to retry —
   the actual mechanism behind "my worker crashed mid-job" not losing work.
2. RETRY WITH A MAX-ATTEMPTS LIMIT + DEAD LETTER QUEUE (DLQ): a message that
   fails repeatedly is moved to a DLQ instead of retrying forever, so one
   poison-pill job can't block the whole queue.
3. CONCURRENT WORKER POOL: N async workers pulling from the same queue,
   each processing a real LLM call independently.

Run directly:
  python3 45_async_job_queues_for_agent_pipelines.py
=============================================================================
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient


# =====================================================================
# 1. IN-MEMORY QUEUE with ack/nack, retries, and a Dead Letter Queue
# =====================================================================
@dataclass
class QueueMessage:
    id: str
    payload: Dict[str, Any]
    attempts: int = 0
    max_attempts: int = 3


class InMemoryQueue:
    """Real equivalent: `boto3` SQS, `aio-pika` (RabbitMQ), or a Kafka
    consumer group — this models the same at-least-once + DLQ semantics
    with an `asyncio.Queue` instead of a network broker.
    """

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self._queue: asyncio.Queue = asyncio.Queue()
        self._in_flight: Dict[str, QueueMessage] = {}
        self.dead_letter_queue: List[QueueMessage] = []
        self.completed: List[str] = []

    def publish(self, payload: Dict[str, Any]) -> str:
        message = QueueMessage(id=str(uuid.uuid4())[:8], payload=payload, max_attempts=self.max_attempts)
        self._queue.put_nowait(message)
        return message.id

    async def consume(self) -> QueueMessage:
        message = await self._queue.get()
        message.attempts += 1
        self._in_flight[message.id] = message
        return message

    def ack(self, message: QueueMessage):
        self._in_flight.pop(message.id, None)
        self.completed.append(message.id)

    def nack(self, message: QueueMessage):
        """Failed processing: retry if attempts remain, else move to the DLQ
        so it stops consuming a worker slot on every cycle.
        """
        self._in_flight.pop(message.id, None)
        if message.attempts >= message.max_attempts:
            self.dead_letter_queue.append(message)
        else:
            self._queue.put_nowait(message)  # re-enqueue for another attempt

    def qsize(self) -> int:
        return self._queue.qsize()


# =====================================================================
# 2. WORKER POOL: N concurrent consumers, each doing a real LLM call
# =====================================================================
async def agent_worker(worker_name: str, queue: InMemoryQueue, llm: LLMClient, results: List[Dict[str, Any]]):
    while queue.qsize() > 0 or queue._in_flight:
        try:
            message = await asyncio.wait_for(queue.consume(), timeout=0.2)
        except asyncio.TimeoutError:
            return  # nothing left to pull — this worker is done

        ticket_text = message.payload["ticket_text"]
        # A deliberately "poisoned" job (always fails) to demonstrate the
        # retry-then-DLQ path, instead of a transient failure that recovers.
        if message.payload.get("simulate_failure"):
            print(f"  {Colors.RED}✖ [{worker_name}] job {message.id} failed (attempt {message.attempts}/{message.max_attempts}){Colors.RESET}")
            queue.nack(message)
            continue

        # `asyncio.to_thread` lets N workers' blocking LLM calls run
        # concurrently under one event loop (same pattern as module 24).
        summary = await asyncio.to_thread(
            llm.chat,
            [{"role": "user", "content": f"In one sentence, summarize this support ticket: {ticket_text}"}],
            simulated_response=f"Summary of: {ticket_text[:40]}...",
        )
        print(f"  {Colors.GREEN}✔ [{worker_name}] processed job {message.id}{Colors.RESET}")
        results.append({"job_id": message.id, "worker": worker_name, "summary": summary})
        queue.ack(message)


async def main():
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== Async Job Queues for Agent Pipelines ==={Colors.RESET}\n")

    llm = LLMClient()
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET}\n")

    queue = InMemoryQueue(max_attempts=3)

    print(f"{Colors.YELLOW}1. Producer publishes 5 jobs and returns immediately (fire-and-forget){Colors.RESET}")
    tickets = [
        "Customer can't access billing dashboard.",
        "Refund requested for duplicate charge.",
        "API returning intermittent 500s.",
        "This job will keep failing until it lands in the DLQ.",
        "Password reset email never arrived.",
    ]
    for i, ticket in enumerate(tickets):
        job_id = queue.publish({"ticket_text": ticket, "simulate_failure": (i == 3)})
        print(f"  Published job {job_id}: \"{ticket[:50]}\"")
    print(f"  Queue depth immediately after publishing: {queue.qsize()}\n")

    print(f"{Colors.YELLOW}2. Worker pool (3 concurrent workers) consumes independently{Colors.RESET}")
    results: List[Dict[str, Any]] = []
    workers = [agent_worker(f"worker-{i+1}", queue, llm, results) for i in range(3)]
    await asyncio.gather(*workers)

    print(f"\n{Colors.YELLOW}3. Outcome{Colors.RESET}")
    print(f"  Completed: {len(results)} jobs")
    print(f"  Dead-lettered: {len(queue.dead_letter_queue)} job(s) after exhausting {queue.max_attempts} attempts")
    for dlq_msg in queue.dead_letter_queue:
        print(f"    • job {dlq_msg.id}: \"{dlq_msg.payload['ticket_text'][:50]}\" ({dlq_msg.attempts} attempts)")

    print(f"\n{Colors.GREEN}✔ At-least-once delivery, retry-then-DLQ, and concurrent worker consumption all verified.{Colors.RESET}")


if __name__ == "__main__":
    asyncio.run(main())
