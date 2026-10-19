"""
HTTP client for communicating with team servers.

Streams a single /ask request, captures all timing events, and returns
parsed events with timestamps.
"""

import time
import httpx
from dataclasses import dataclass, field
from typing import List, Optional

from harness.contract import (
    Event, RetrievedEvent, AudioEvent, DoneEvent,
    parse_event, validate_event_sequence, AskRequest,
    audio_duration_seconds, decode_audio_chunk
)


@dataclass
class TimedEvent:
    """An event with its receipt timestamp (from time.perf_counter())."""
    event: Event
    timestamp: float  # perf_counter() value when this line was fully received


@dataclass
class AskResult:
    """Result of a single /ask request."""
    question_id: str
    events: List[TimedEvent] = field(default_factory=list)
    error: Optional[str] = None
    success: bool = True
    
    # Derived timing metrics (populated after stream completes)
    t_send: float = 0.0
    t_retrieved: float = 0.0
    t_first_audio: float = 0.0
    t_done: float = 0.0
    
    # Stall simulation metrics
    stall_count: int = 0
    total_stall_ms: float = 0.0

    def compute_derived_metrics(self, stall_min_ms: float = 200.0) -> None:
        """
        Compute all derived timing metrics from the event timestamps.
        
        Args:
            stall_min_ms: Minimum stall duration in ms to count as a stall
        """
        if not self.success or not self.events:
            return
        
        # Find key timestamps
        for timed_event in self.events:
            if isinstance(timed_event.event, RetrievedEvent):
                self.t_retrieved = timed_event.timestamp
                break
        
        for timed_event in self.events:
            if isinstance(timed_event.event, AudioEvent) and timed_event.event.data:
                self.t_first_audio = timed_event.timestamp
                break
        
        for timed_event in self.events:
            if isinstance(timed_event.event, DoneEvent):
                self.t_done = timed_event.timestamp
                break
        
        # Validate we have all required events
        has_retrieved = any(isinstance(te.event, RetrievedEvent) for te in self.events)
        has_audio = any(isinstance(te.event, AudioEvent) and te.event.data for te in self.events)
        has_done = any(isinstance(te.event, DoneEvent) for te in self.events)
        
        if not (has_retrieved and has_audio and has_done):
            self.success = False
            self.error = "Missing required event(s): " + \
                ", ".join([
                    "retrieved" if not has_retrieved else "",
                    "audio" if not has_audio else "",
                    "done" if not has_done else ""
                ]).strip(", ")
            return
        
        # Compute stall simulation
        self._compute_stalls(stall_min_ms)
    
    def _compute_stalls(self, stall_min_ms: float) -> None:
        """
        Simulate playback stall detection.
        
        A player starts when first audio chunk arrives and plays in real time.
        Track buffered_end (when buffered audio runs out). When a chunk arrives
        at time t and t > buffered_end, the player starved.
        """
        if self.t_first_audio == 0:
            return
        
        buffered_end = self.t_first_audio  # player starts at first audio
        stall_threshold_s = stall_min_ms / 1000.0
        
        for timed_event in self.events:
            if not isinstance(timed_event.event, AudioEvent):
                continue
            if not timed_event.event.data:
                continue
            
            chunk_arrival = timed_event.timestamp
            chunk_bytes = decode_audio_chunk(timed_event.event.data)
            chunk_duration = audio_duration_seconds(chunk_bytes)
            
            # Check for stall: player starved if chunk arrives after buffered_end
            if chunk_arrival > buffered_end:
                stall_gap = chunk_arrival - buffered_end
                self.total_stall_ms += stall_gap * 1000.0
                if stall_gap >= stall_threshold_s:
                    self.stall_count += 1
            
            # Update buffered_end: max(current time, buffered_end) + chunk duration
            buffered_end = max(chunk_arrival, buffered_end) + chunk_duration


class TeamClient:
    """
    HTTP client for a team's server.
    
    Streams the /ask endpoint and captures all timing information.
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        connect_timeout: float = 5.0
    ):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the team server (e.g., "http://localhost:8000")
            timeout: Total request timeout in seconds
            connect_timeout: Connection timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.connect_timeout = connect_timeout
    
    def health_check(self) -> bool:
        """Check if the team server is healthy via GET /health."""
        try:
            with httpx.Client(timeout=self.connect_timeout) as client:
                resp = client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
    
    def ask(self, request: AskRequest) -> AskResult:
        """
        Send a question to the team's /ask endpoint and stream the response.
        
        Args:
            request: The question to ask
            
        Returns:
            AskResult with all events, timestamps, and derived metrics
        """
        result = AskResult(question_id=request.question_id)
        url = f"{self.base_url}/ask"
        
        # t_send: just before sending the request
        result.t_send = time.perf_counter()
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    url,
                    json=request.to_json(),
                    headers={"Accept": "application/x-ndjson"}
                ) as response:
                    response.raise_for_status()
                    
                    # Stream and parse NDJSON lines
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        timestamp = time.perf_counter()
                        event = parse_event(line)
                        
                        if event is not None:
                            result.events.append(TimedEvent(event=event, timestamp=timestamp))
        
        except httpx.TimeoutException:
            result.success = False
            result.error = f"Request timed out after {self.timeout}s"
        except httpx.HTTPStatusError as e:
            result.success = False
            result.error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            result.success = False
            result.error = f"Request failed: {type(e).__name__}: {e}"
        
        # Validate event sequence
        if result.success:
            events = [te.event for te in result.events]
            errors = validate_event_sequence(events)
            if errors:
                result.success = False
                result.error = "Event validation failed: " + "; ".join(errors)
        
        # Compute derived metrics if successful
        if result.success:
            result.compute_derived_metrics()
        else:
            # Ensure timing fields are None for failed questions
            result.t_retrieved = 0.0
            result.t_first_audio = 0.0
            result.t_done = 0.0
        
        return result


def create_client(base_url: str, config: dict) -> TeamClient:
    """Factory function to create a client from config."""
    return TeamClient(
        base_url=base_url,
        timeout=config.get("request_timeout_s", 30),
        connect_timeout=5.0
    )
