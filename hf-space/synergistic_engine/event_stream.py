from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time


class EventType(str, Enum):
    QUERY = "QUERY"
    CONTEXT_RETRIEVED = "CONTEXT_RETRIEVED"
    TOOL_CALLED = "TOOL_CALLED"
    RESPONSE_GENERATED = "RESPONSE_GENERATED"
    ERROR = "ERROR"
    THRESHOLD_EXCEEDED = "THRESHOLD_EXCEEDED"
    PATTERN_MATCHED = "PATTERN_MATCHED"
    CUSTOM = "CUSTOM"


@dataclass
class Event:
    type: EventType
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = "system"

    def matches(self, type_filter: EventType | None = None, **payload_filters) -> bool:
        if type_filter and self.type != type_filter:
            return False
        for k, v in payload_filters.items():
            if self.payload.get(k) != v:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
        }


class EventStream:
    def __init__(self, window_size: int = 20):
        self._events: list[Event] = []
        self.window_size = window_size

    def emit(self, event: Event) -> None:
        self._events.append(event)
        if len(self._events) > self.window_size:
            self._events.pop(0)

    def recent(self, n: int | None = None) -> list[Event]:
        return self._events[-n:] if n else self._events[:]

    def count_type(self, etype: EventType, last_n: int = 10) -> int:
        return sum(1 for e in self._events[-last_n:] if e.type == etype)

    def last_of_type(self, etype: EventType) -> Event | None:
        for e in reversed(self._events):
            if e.type == etype:
                return e
        return None

    def clear(self) -> None:
        self._events.clear()
