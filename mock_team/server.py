"""
Mock team server for testing the evaluation harness.

Implements the submission contract with four modes:
- good: keyword-based retrieval, progressive streaming with small chunks
- nonstreaming: same retrieval, but sends all audio at once after delay
- stalling: progressive streaming but inserts a 700ms pause midway
- badretrieval: returns random passage IDs
"""

import asyncio
import base64
import json
import math
import random
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# =============================================================================
# Audio Constants (must match harness/contract.py)
# =============================================================================
AUDIO_SAMPLE_RATE = 24000
AUDIO_BIT_DEPTH = 16
AUDIO_CHANNELS = 1
AUDIO_BYTES_PER_SAMPLE = AUDIO_BIT_DEPTH // 8
AUDIO_BYTES_PER_SECOND = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_BYTES_PER_SAMPLE


def generate_silence(duration_s: float) -> bytes:
    """Generate silent PCM audio (zeros)."""
    num_samples = int(duration_s * AUDIO_SAMPLE_RATE)
    return b"\x00\x00" * num_samples


@lru_cache(maxsize=32)
def generate_sine_tone(duration_s: float, frequency: float = 440.0) -> bytes:
    """Generate a sine tone as 16-bit little-endian mono PCM.

    Cached so repeated chunk generation doesn't burn CPU and skew
    latency measurements in the harness.
    """
    num_samples = int(duration_s * AUDIO_SAMPLE_RATE)
    samples = [
        max(-32768, min(32767, int(32767 * 0.1 * math.sin(2 * math.pi * frequency * i / AUDIO_SAMPLE_RATE))))
        for i in range(num_samples)
    ]
    return struct.pack(f"<{num_samples}h", *samples)


def audio_duration_seconds(pcm_bytes: bytes) -> float:
    """Calculate audio duration in seconds from raw PCM bytes."""
    return len(pcm_bytes) / AUDIO_BYTES_PER_SECOND


def encode_audio_chunk(pcm_bytes: bytes) -> str:
    """Encode PCM audio chunk to base64 string."""
    return base64.b64encode(pcm_bytes).decode("ascii")


def ndjson(obj: dict) -> str:
    """Serialize one NDJSON line (valid JSON + trailing newline)."""
    return json.dumps(obj) + "\n"


# =============================================================================
# Sample Data (embedded for self-contained server)
# =============================================================================
PASSAGES = [
    {"passage_id": "p01", "doc": "leave_policy.pdf", "page": 4, "text": "Employees are entitled to 20 days of annual leave per calendar year. Leave must be requested at least 2 weeks in advance through the HR portal."},
    {"passage_id": "p02", "doc": "leave_policy.pdf", "page": 5, "text": "Sick leave is unlimited with a doctor's note for absences longer than 3 consecutive days. Short-term sick leave (1-3 days) requires only manager notification."},
    {"passage_id": "p03", "doc": "leave_policy.pdf", "page": 6, "text": "Parental leave provides 12 weeks paid leave for primary caregivers and 4 weeks for secondary caregivers, available within the first year of birth or adoption."},
    {"passage_id": "p04", "doc": "travel_policy.pdf", "page": 2, "text": "Business travel must be pre-approved by a director. Economy class is standard for flights under 6 hours; business class requires VP approval."},
    {"passage_id": "p05", "doc": "travel_policy.pdf", "page": 3, "text": "Meal per diem is $75/day domestic, $100/day international. Receipts required for expenses over $25. Alcohol is not reimbursable."},
    {"passage_id": "p06", "doc": "expense_policy.pdf", "page": 1, "text": "All expenses must be submitted within 30 days of incurred date. Corporate card preferred; personal card reimbursements require itemized receipts."},
    {"passage_id": "p07", "doc": "expense_policy.pdf", "page": 2, "text": "Ground transportation: use rideshare or public transit. Rental cars require prior approval. Parking and tolls are reimbursable with receipts."},
    {"passage_id": "p08", "doc": "remote_work_policy.pdf", "page": 1, "text": "Employees may work remotely up to 3 days per week with manager approval. Full remote requires director approval and annual review."},
    {"passage_id": "p09", "doc": "remote_work_policy.pdf", "page": 2, "text": "Remote workers must be available during core hours 10am-3pm ET. Company provides $500 annual stipend for home office equipment."},
    {"passage_id": "p10", "doc": "code_of_conduct.pdf", "page": 1, "text": "All employees must treat colleagues with respect. Harassment, discrimination, and retaliation are prohibited. Report violations to HR or ethics hotline."},
]

PASSAGE_IDS = [p["passage_id"] for p in PASSAGES]


# =============================================================================
# Retrieval Logic
# =============================================================================
def keyword_retrieve(question: str, k: int = 10) -> List[str]:
    """Simple keyword-based retrieval."""
    question_words = set(question.lower().split())
    scores = []

    for p in PASSAGES:
        text_words = set(p["text"].lower().split())
        overlap = len(question_words & text_words)
        scores.append((overlap, p["passage_id"]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [pid for _, pid in scores[:k]]


def random_retrieve(k: int = 10) -> List[str]:
    """Return random passage IDs."""
    return random.sample(PASSAGE_IDS, min(k, len(PASSAGE_IDS)))


# =============================================================================
# Request/Response Models
# =============================================================================
class AskRequest(BaseModel):
    question_id: str
    question: str


class HealthResponse(BaseModel):
    status: str = "ok"


# =============================================================================
# Server Modes
# =============================================================================
class ServerMode:
    GOOD = "good"
    NONSTREAMING = "nonstreaming"
    STALLING = "stalling"
    BADRETRIEVAL = "badretrieval"


VALID_MODES = (
    ServerMode.GOOD,
    ServerMode.NONSTREAMING,
    ServerMode.STALLING,
    ServerMode.BADRETRIEVAL,
)


@dataclass
class ServerConfig:
    mode: str = ServerMode.GOOD
    retrieval_delay_ms: int = 150
    chunk_duration_ms: int = 100
    inter_chunk_delay_ms: int = 80
    stall_duration_ms: int = 700
    total_audio_duration_s: float = 3.0


async def _stream_audio(config: ServerConfig, stall: bool = False):
    """Progressively stream small audio chunks, optionally stalling midway."""
    total_chunks = int(config.total_audio_duration_s * 1000 / config.chunk_duration_ms)
    stall_at = total_chunks // 2

    for i in range(total_chunks):
        pcm = generate_sine_tone(config.chunk_duration_ms / 1000.0)
        yield ndjson({"type": "audio", "data": encode_audio_chunk(pcm)})

        if stall and i == stall_at:
            await asyncio.sleep(config.stall_duration_ms / 1000.0)
        else:
            await asyncio.sleep(config.inter_chunk_delay_ms / 1000.0)

    yield ndjson({"type": "done"})


async def stream_good(question: str, config: ServerConfig):
    """Good mode: progressive streaming with small chunks."""
    yield ndjson({"type": "retrieved", "passage_ids": keyword_retrieve(question)})
    await asyncio.sleep(config.retrieval_delay_ms / 1000.0)
    async for line in _stream_audio(config):
        yield line


async def stream_nonstreaming(question: str, config: ServerConfig):
    """Nonstreaming mode: all audio at once after delay."""
    yield ndjson({"type": "retrieved", "passage_ids": keyword_retrieve(question)})
    await asyncio.sleep(3.0)
    pcm = generate_sine_tone(config.total_audio_duration_s)
    yield ndjson({"type": "audio", "data": encode_audio_chunk(pcm)})
    yield ndjson({"type": "done"})


async def stream_stalling(question: str, config: ServerConfig):
    """Stalling mode: progressive but with a pause midway."""
    yield ndjson({"type": "retrieved", "passage_ids": keyword_retrieve(question)})
    await asyncio.sleep(config.retrieval_delay_ms / 1000.0)
    async for line in _stream_audio(config, stall=True):
        yield line


async def stream_badretrieval(question: str, config: ServerConfig):
    """Bad retrieval mode: random passages, but streams audio properly."""
    yield ndjson({"type": "retrieved", "passage_ids": random_retrieve()})
    await asyncio.sleep(config.retrieval_delay_ms / 1000.0)
    async for line in _stream_audio(config):
        yield line


STREAMERS = {
    ServerMode.GOOD: stream_good,
    ServerMode.NONSTREAMING: stream_nonstreaming,
    ServerMode.STALLING: stream_stalling,
    ServerMode.BADRETRIEVAL: stream_badretrieval,
}


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(title="Mock Team Server")

server_config = ServerConfig()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/ask")
async def ask(request: AskRequest):
    """Stream NDJSON response."""
    # Validate before streaming starts so the client actually receives a 400.
    streamer = STREAMERS.get(server_config.mode)
    if streamer is None:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {server_config.mode}")

    return StreamingResponse(
        streamer(request.question, server_config),
        media_type="application/x-ndjson",
    )


def create_app(mode: str = ServerMode.GOOD) -> FastAPI:
    """Create app with specific mode."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    server_config.mode = mode
    return app


if __name__ == "__main__":
    import sys

    mode = ServerMode.GOOD
    port = 8000

    if len(sys.argv) > 1:
        if sys.argv[1] in VALID_MODES:
            mode = sys.argv[1]
        else:
            print("Usage: python server.py [good|nonstreaming|stalling|badretrieval] [port]")
            sys.exit(1)

    if len(sys.argv) > 2:
        port = int(sys.argv[2])

    server_config.mode = mode
    print(f"Starting mock server in '{mode}' mode on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")