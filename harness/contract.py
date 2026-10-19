"""
Contract definitions for the voice-driven RAG assistant evaluation harness.

Defines audio format constants, NDJSON event types, and validation utilities.
"""

import json
import base64
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


# =============================================================================
# Audio Format Constants (defined in one place as required)
# =============================================================================
AUDIO_SAMPLE_RATE = 24000          # Hz
AUDIO_BIT_DEPTH = 16               # bits per sample (16-bit signed)
AUDIO_CHANNELS = 1                 # mono
AUDIO_BYTES_PER_SAMPLE = AUDIO_BIT_DEPTH // 8  # 2 bytes
AUDIO_BYTES_PER_SECOND = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_BYTES_PER_SAMPLE  # 48000 bytes/sec


def audio_duration_seconds(pcm_bytes: bytes) -> float:
    """
    Calculate audio duration in seconds from raw PCM bytes.
    
    Args:
        pcm_bytes: Raw PCM audio data (16-bit signed little-endian, mono, 24000 Hz)
        
    Returns:
        Duration in seconds
    """
    return len(pcm_bytes) / AUDIO_BYTES_PER_SECOND


def decode_audio_chunk(b64_data: str) -> bytes:
    """Decode base64-encoded PCM audio chunk."""
    return base64.b64decode(b64_data)


def encode_audio_chunk(pcm_bytes: bytes) -> str:
    """Encode PCM audio chunk to base64 string."""
    return base64.b64encode(pcm_bytes).decode('ascii')


# =============================================================================
# NDJSON Event Types
# =============================================================================
class EventType(str, Enum):
    RETRIEVED = "retrieved"
    AUDIO = "audio"
    DONE = "done"


@dataclass
class RetrievedEvent:
    """Event sent once with ranked passage IDs."""
    type: str = EventType.RETRIEVED
    passage_ids: List[str] = None
    
    def __post_init__(self):
        if self.passage_ids is None:
            self.passage_ids = []


@dataclass
class AudioEvent:
    """Event sent zero or more times with base64-encoded PCM audio."""
    type: str = EventType.AUDIO
    data: str = ""


@dataclass
class DoneEvent:
    """Event sent once at the end of the stream."""
    type: str = EventType.DONE


Event = RetrievedEvent | AudioEvent | DoneEvent


# =============================================================================
# Event Parsing & Validation
# =============================================================================
def parse_event(line: str) -> Optional[Event]:
    """
    Parse a single NDJSON line into an Event.
    
    Returns None for empty lines or invalid JSON.
    """
    line = line.strip()
    if not line:
        return None
    
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    
    event_type = obj.get("type")
    if event_type == EventType.RETRIEVED:
        return RetrievedEvent(
            type=event_type,
            passage_ids=obj.get("passage_ids", [])
        )
    elif event_type == EventType.AUDIO:
        return AudioEvent(
            type=event_type,
            data=obj.get("data", "")
        )
    elif event_type == EventType.DONE:
        return DoneEvent(type=event_type)
    
    return None


def validate_event_sequence(events: List[Event]) -> List[str]:
    """
    Validate a sequence of events follows the contract.
    
    Returns list of validation errors (empty if valid).
    Rules:
    - Exactly one 'retrieved' event (first non-audio event)
    - Zero or more 'audio' events
    - Exactly one 'done' event (last)
    - 'retrieved' must come before any 'audio' or 'done'
    - 'done' must be the last event
    """
    errors = []
    
    retrieved_count = 0
    done_count = 0
    seen_retrieved = False
    seen_done = False
    
    for i, event in enumerate(events):
        if event.type == EventType.RETRIEVED:
            retrieved_count += 1
            if seen_done:
                errors.append(f"Event {i}: 'retrieved' after 'done'")
            if i != 0:
                errors.append(f"Event {i}: 'retrieved' must be first event")
            seen_retrieved = True
            
        elif event.type == EventType.AUDIO:
            if not seen_retrieved:
                errors.append(f"Event {i}: 'audio' before 'retrieved'")
            if seen_done:
                errors.append(f"Event {i}: 'audio' after 'done'")
                
        elif event.type == EventType.DONE:
            done_count += 1
            if not seen_retrieved:
                errors.append(f"Event {i}: 'done' before 'retrieved'")
            seen_done = True
    
    if retrieved_count != 1:
        errors.append(f"Expected exactly 1 'retrieved' event, got {retrieved_count}")
    if done_count != 1:
        errors.append(f"Expected exactly 1 'done' event, got {done_count}")
    
    return errors


# =============================================================================
# Request/Response Models
# =============================================================================
@dataclass
class AskRequest:
    """Request to the team's /ask endpoint."""
    question_id: str
    question: str
    
    def to_json(self) -> dict:
        return {"question_id": self.question_id, "question": self.question}


@dataclass
class HealthResponse:
    """Response from the team's /health endpoint."""
    status: str = "ok"
    
    def to_json(self) -> dict:
        return {"status": self.status}