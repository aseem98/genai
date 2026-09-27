"""
=============================================================================
07_multimodal_vision_inputs.py: Images In, Structured Answers Out
=============================================================================
Modern LLMs are multimodal: the same chat endpoint accepts images alongside
text. This is how production systems read scanned documents, charts, and
screenshots. This module:

1. GENERATES REAL IMAGES in pure Python — a minimal PNG encoder (zlib +
   CRC-framed chunks, the actual PNG file format) plus a 5x7 bitmap font to
   render text into pixels, so we control the exact ground truth.
2. SENDS THEM using the OpenAI-compatible multimodal message format: a
   content LIST of {"type": "text"} and {"type": "image_url"} parts, with
   the image inlined as a base64 data URL.
3. SCORES THE ANSWERS against that ground truth:
   - OCR of a rendered "scanned invoice" into structured JSON fields,
   - chart understanding (which bar is tallest, and the full ranking).
4. MEASURES THE COST: an image consumes prompt tokens too — compared below
   against an equivalent text-only request using real `usage` numbers.

The generated PNGs are saved to ./generated_assets/ so you can open them.

Run directly:
  python3 07_multimodal_vision_inputs.py
=============================================================================
"""

import base64
import json
import os
import re
import struct
import zlib
from typing import Dict, List, Tuple
from importlib import import_module

llm_module = import_module("01_unified_llm_client")
Colors = llm_module.Colors
LLMClient = llm_module.LLMClient

Pixel = Tuple[int, int, int]
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_assets")

# 5x7 bitmap font: each glyph is 7 rows of 5 bits (1 = ink).
FONT_5X7: Dict[str, List[str]] = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "$": ["00100", "01111", "10100", "01110", "00101", "11110", "00100"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    " ": ["00000"] * 7,
}


# =====================================================================
# 1. A REAL PNG ENCODER + TEXT RENDERER
# =====================================================================
class Canvas:
    def __init__(self, width: int, height: int, background: Pixel = (255, 255, 255)):
        self.width, self.height = width, height
        self.pixels = [[background] * width for _ in range(height)]

    def fill_rect(self, x0: int, y0: int, w: int, h: int, color: Pixel):
        for y in range(max(0, y0), min(self.height, y0 + h)):
            for x in range(max(0, x0), min(self.width, x0 + w)):
                self.pixels[y][x] = color

    def draw_text(self, text: str, x0: int, y0: int, scale: int = 3, color: Pixel = (20, 20, 20)):
        for i, ch in enumerate(text.upper()):
            glyph = FONT_5X7.get(ch, FONT_5X7[" "])
            for row, bits in enumerate(glyph):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        self.fill_rect(x0 + (i * 6 + col) * scale, y0 + row * scale, scale, scale, color)

    def to_png(self) -> bytes:
        """PNG = 8-byte signature + IHDR (size, 8-bit RGB) + IDAT (zlib of
        rows, each prefixed with filter byte 0) + IEND, each chunk framed
        as length | type | data | CRC32."""
        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in self.pixels)
        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def image_message(prompt: str, png: bytes) -> List[Dict]:
    data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}]


def parse_json_object(text: str) -> Dict:
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1]) if start != -1 and end > start else {}
    except json.JSONDecodeError:
        return {}


def to_number(value) -> float:
    """Models return totals as 1250.75, "1250.75" or "$1,250.75" — normalize all three."""
    return float(re.sub(r"[^\d.\-]", "", str(value)))


def ask_about_image(llm: LLMClient, prompt: str, png: bytes, simulated_response: str, max_attempts: int = 2) -> Dict:
    """Validate-and-retry: a multi-model gateway route can silently drop the
    image part, and the model then (correctly!) reports it sees nothing —
    an all-null/empty answer. Retrying once usually lands on a working route;
    pinning VISION_MODEL to a vision-capable model avoids the issue entirely."""
    for attempt in range(1, max_attempts + 1):
        parsed = parse_json_object(llm.chat(image_message(prompt, png), simulated_response=simulated_response, temperature=0))
        if parsed and any(v not in (None, "", []) for v in parsed.values()):
            return parsed
        print(f"  {Colors.YELLOW}⚠ attempt {attempt}: empty answer — the route likely dropped the image; retrying...{Colors.RESET}")
    return parsed


def save(name: str, png: bytes) -> str:
    os.makedirs(ASSET_DIR, exist_ok=True)
    path = os.path.join(ASSET_DIR, name)
    with open(path, "wb") as f:
        f.write(png)
    return path


if __name__ == "__main__":
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== 07: Multimodal Vision Inputs ==={Colors.RESET}\n")
    llm = LLMClient(model=os.getenv("VISION_MODEL") or None)
    print(f"Active Mode: {Colors.GREEN}{llm.mode.upper()}{Colors.RESET} | model: {llm.model}"
          f"{'' if os.getenv('VISION_MODEL') else '  (set VISION_MODEL in .env to pin a vision-capable model)'}\n")

    # ---- 1. OCR a rendered "scanned invoice" -------------------------------
    truth = {"invoice_number": "INV-4821", "date": "2026-03-14", "vendor": "ACME MEDICAL SUPPLY", "total": 1250.75}
    canvas = Canvas(480, 170, background=(248, 246, 240))           # off-white, like a scan
    canvas.draw_text(truth["vendor"], 20, 15, scale=3)
    canvas.draw_text(f"INVOICE NO: {truth['invoice_number']}", 20, 55, scale=2)
    canvas.draw_text(f"DATE: {truth['date']}", 20, 80, scale=2)
    canvas.fill_rect(20, 105, 440, 2, (120, 120, 120))
    canvas.draw_text(f"TOTAL DUE: ${truth['total']:.2f}", 20, 120, scale=3)
    invoice_png = canvas.to_png()
    print(f"{Colors.YELLOW}1. OCR -> structured JSON from a rendered invoice image{Colors.RESET}")
    print(f"  Wrote {len(invoice_png):,}-byte PNG: {save('invoice.png', invoice_png)}")

    extracted = ask_about_image(
        llm, "Extract these fields from the invoice image as JSON with keys "
             "invoice_number, date (YYYY-MM-DD), vendor, total (a number). Return only the JSON.",
        invoice_png, simulated_response=json.dumps(truth),
    )
    image_prompt_tokens = (llm.last_usage or {}).get("prompt_tokens")
    correct = 0
    for field, expected in truth.items():
        got = extracted.get(field)
        if field == "total":
            try:
                ok = got is not None and abs(to_number(got) - expected) < 0.01
            except ValueError:
                ok = False
        else:
            ok = str(got).strip().upper() == str(expected).upper()
        correct += ok
        print(f"    {field:<15} expected={expected!s:<22} got={got!s:<22} {Colors.GREEN + '✔' if ok else Colors.RED + '✖'}{Colors.RESET}")
    print(f"  Field accuracy: {correct}/{len(truth)}\n")

    # ---- 2. Chart understanding --------------------------------------------
    bars = [("RED", 70, (220, 50, 50)), ("BLUE", 140, (50, 90, 220)), ("GREEN", 105, (40, 160, 70)), ("ORANGE", 35, (240, 150, 30))]
    chart = Canvas(360, 200)
    chart.fill_rect(20, 180, 320, 2, (0, 0, 0))                     # x-axis
    for i, (_, height, color) in enumerate(bars):
        chart.fill_rect(40 + i * 75, 180 - height, 50, height, color)
    chart_png = chart.to_png()
    expected_ranking = [name for name, _, _ in sorted(bars, key=lambda b: b[1], reverse=True)]
    print(f"{Colors.YELLOW}2. Chart understanding{Colors.RESET}")
    print(f"  Wrote chart PNG: {save('bar_chart.png', chart_png)}")
    answer = ask_about_image(
        llm, "This bar chart has four bars colored red, blue, green and orange. List the colors from tallest bar to "
             "shortest as a JSON object {\"ranking\": [...]} using uppercase color names. Return only the JSON.",
        chart_png, simulated_response=json.dumps({"ranking": expected_ranking}),
    )
    ranking = [str(c).upper() for c in answer.get("ranking", [])]
    print(f"  expected: {expected_ranking}")
    print(f"  model:    {ranking}  {Colors.GREEN + '✔ exact match' if ranking == expected_ranking else Colors.RED + '✖ mismatch'}{Colors.RESET}\n")

    # ---- 3. What an image costs -------------------------------------------
    print(f"{Colors.YELLOW}3. Token cost of an image vs. the same information as text{Colors.RESET}")
    llm.chat([{"role": "user", "content": f"Extract invoice fields as JSON from this text: {json.dumps(truth)}"}],
             simulated_response="{}", max_tokens=5)
    text_prompt_tokens = (llm.last_usage or {}).get("prompt_tokens")
    if image_prompt_tokens and text_prompt_tokens:
        print(f"  image request prompt_tokens={image_prompt_tokens} | text-only prompt_tokens={text_prompt_tokens}")
        print(f"  {Colors.DIM}If the text is already available (a digital PDF), sending text is cheaper than sending pixels.{Colors.RESET}")
    else:
        print("  (real token counts need live mode)")

    print(f"\n{Colors.GREEN}✔ Real PNGs generated, sent as multimodal messages, and answers scored against ground truth.{Colors.RESET}")
