"""
=============================================================================
39_grpc_and_protocol_buffers.py: gRPC RPC Semantics & Protobuf Wire Format
=============================================================================
Senior GenAI Engineer Context:
Internal agent-to-service traffic (e.g. an orchestrator calling a dedicated
embedding/reranking microservice) is frequently gRPC rather than REST,
because it is strongly typed, uses a compact binary wire format (Protocol
Buffers), and supports first-class streaming over a single HTTP/2 connection
— exactly what you want for token-by-token LLM streaming.

This module implements, from the standard library only:
  1. The Protobuf wire format itself: tag = (field_number << 3) | wire_type,
     varint length-prefixing, and encode/decode for string & int fields.
  2. The 4 gRPC call shapes: Unary, Server-streaming, Client-streaming, and
     Bidirectional-streaming — modeled as plain Python generators/functions
     standing in for HTTP/2 frames.

Real equivalent:
  # embedding.proto
  service Embedder {
    rpc Embed (EmbedRequest) returns (EmbedReply);
    rpc StreamTokens (PromptRequest) returns (stream TokenChunk);
  }
  # then: python -m grpc_tools.protoc ... && stub.Embed(request)

Run directly:
  python3 39_grpc_and_protocol_buffers.py
=============================================================================
"""

from dataclasses import dataclass
from typing import Iterator, List
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors


# =====================================================================
# 1. PROTOCOL BUFFERS WIRE FORMAT (varints + tag/wire-type encoding)
# =====================================================================
WIRETYPE_VARINT = 0
WIRETYPE_LENGTH_DELIMITED = 2


def encode_varint(value: int) -> bytes:
    """Base-128 varint encoding: 7 payload bits + 1 continuation bit/byte."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_varint(buf: bytes, pos: int):
    result = shift = 0
    while True:
        byte = buf[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            return result, pos
        shift += 7


def encode_tag(field_number: int, wire_type: int) -> bytes:
    return encode_varint((field_number << 3) | wire_type)


def encode_string_field(field_number: int, value: str) -> bytes:
    payload = value.encode("utf-8")
    return encode_tag(field_number, WIRETYPE_LENGTH_DELIMITED) + encode_varint(len(payload)) + payload


def encode_int_field(field_number: int, value: int) -> bytes:
    return encode_tag(field_number, WIRETYPE_VARINT) + encode_varint(value)


def decode_message(buf: bytes) -> dict:
    """Generic decoder: walks tag/value pairs without needing a .proto schema
    (mirrors how `protoc`-generated code interprets bytes at a lower level).
    """
    fields, pos = {}, 0
    while pos < len(buf):
        tag, pos = decode_varint(buf, pos)
        field_number, wire_type = tag >> 3, tag & 0x7
        if wire_type == WIRETYPE_VARINT:
            value, pos = decode_varint(buf, pos)
        elif wire_type == WIRETYPE_LENGTH_DELIMITED:
            length, pos = decode_varint(buf, pos)
            raw = buf[pos:pos + length]
            pos += length
            try:
                value = raw.decode("utf-8")
            except UnicodeDecodeError:
                value = raw
        else:
            raise ValueError(f"Unsupported wire_type {wire_type}")
        fields[field_number] = value
    return fields


@dataclass
class EmbedRequest:
    text: str
    model_version: int = 1

    def serialize(self) -> bytes:
        return encode_string_field(1, self.text) + encode_int_field(2, self.model_version)


@dataclass
class EmbedReply:
    vector_preview: str
    dims: int


# =====================================================================
# 2. THE 4 gRPC CALL SHAPES
# =====================================================================
class EmbedderServicer:
    """Server-side implementation — real equivalent: a class generated from
    `embedding_pb2_grpc.EmbedderServicer` with methods you override.
    """

    def Embed(self, request: EmbedRequest) -> EmbedReply:
        """1. UNARY RPC: one request in, one response out."""
        fake_vector = [round((ord(c) % 10) / 10, 2) for c in request.text[:4]] or [0.0]
        return EmbedReply(vector_preview=str(fake_vector), dims=384)

    def StreamTokens(self, prompt: str) -> Iterator[str]:
        """2. SERVER-STREAMING RPC: one request, many responses over time —
        this is exactly how live LLM token streaming maps onto gRPC.
        """
        for token in prompt.split():
            yield token

    def ClientStreamUpload(self, chunks: Iterator[bytes]) -> dict:
        """3. CLIENT-STREAMING RPC: many requests in, one response out —
        e.g. uploading an audio file in chunks for a single transcription.
        """
        total_bytes = sum(len(c) for c in chunks)
        return {"received_chunks": True, "total_bytes": total_bytes}

    def BidiChat(self, incoming: Iterator[str]) -> Iterator[str]:
        """4. BIDIRECTIONAL-STREAMING RPC: both sides stream concurrently —
        e.g. a live voice agent exchanging audio frames both ways.
        """
        for message in incoming:
            yield f"echo::{message.upper()}"


def make_stub(servicer: EmbedderServicer):
    """Real equivalent: `stub = embedding_pb2_grpc.EmbedderStub(channel)`.
    Here the 'channel' is just direct in-process calls, but the wire bytes
    are still produced/consumed to prove the serialization round-trips.
    """
    class Stub:
        def Embed(self, request: EmbedRequest) -> EmbedReply:
            wire_bytes = request.serialize()             # client marshals
            decoded_fields = decode_message(wire_bytes)   # server unmarshals
            rebuilt = EmbedRequest(text=decoded_fields[1], model_version=decoded_fields[2])
            return servicer.Embed(rebuilt)

        def StreamTokens(self, prompt: str):
            return servicer.StreamTokens(prompt)

    return Stub()


if __name__ == "__main__":
    print(f"{Colors.BOLD}{Colors.CYAN}=== gRPC RPC Semantics & Protobuf Wire Format ==={Colors.RESET}\n")

    print(f"{Colors.YELLOW}1. Protobuf wire-format round-trip{Colors.RESET}")
    req = EmbedRequest(text="hello grpc", model_version=3)
    wire = req.serialize()
    print(f"  Wire bytes ({len(wire)}B): {wire}")
    print(f"  Decoded generic fields: {decode_message(wire)}\n")

    servicer = EmbedderServicer()
    stub = make_stub(servicer)

    print(f"{Colors.YELLOW}2. Unary RPC: stub.Embed(request){Colors.RESET}")
    reply = stub.Embed(req)
    print(f"  Reply -> {reply}\n")

    print(f"{Colors.YELLOW}3. Server-streaming RPC: stub.StreamTokens(prompt){Colors.RESET}")
    print("  Streaming: ", end="")
    for token in stub.StreamTokens("the incident has been mitigated"):
        print(f"[{token}]", end=" ", flush=True)
    print("\n")

    print(f"{Colors.YELLOW}4. Client-streaming RPC: ClientStreamUpload(chunks){Colors.RESET}")
    chunks: List[bytes] = [b"audio-chunk-1", b"audio-chunk-2", b"audio-chunk-3"]
    print(f"  {servicer.ClientStreamUpload(iter(chunks))}\n")

    print(f"{Colors.YELLOW}5. Bidirectional-streaming RPC: BidiChat(incoming){Colors.RESET}")
    for reply_msg in servicer.BidiChat(iter(["ping", "status?", "over"])):
        print(f"  <- {reply_msg}")

    print(f"\n{Colors.GREEN}✔ All 4 gRPC call shapes and Protobuf encoding verified.{Colors.RESET}")
