from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
import sys


@dataclass(slots=True)
class ProgressEvent:
    domain: str
    stage: str
    current: int | None = None
    total: int | None = None
    elapsed_s: float | None = None
    eta_s: float | None = None
    context: dict[str, Any] = field(default_factory=dict)


class ProgressReporter:
    def emit(self, event: ProgressEvent) -> None:
        raise NotImplementedError


class NullProgressReporter(ProgressReporter):
    def emit(self, event: ProgressEvent) -> None:
        return None


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 1.0:
        return "<1s"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    rounded_seconds = int(round(seconds))
    if rounded_seconds < 3600:
        minutes, secs = divmod(rounded_seconds, 60)
        return f"{minutes}m{secs:02d}s"
    total_minutes = rounded_seconds // 60
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h{minutes:02d}m"


def format_percent(current: int | None, total: int | None) -> str | None:
    if current is None or total is None or total <= 0:
        return None
    percent = max(0.0, min(100.0, 100.0 * float(current) / float(total)))
    return f"{int(round(percent))}%"


class PhaseTimer:
    def __init__(self) -> None:
        self.started_at = perf_counter()

    def elapsed(self) -> float:
        return perf_counter() - self.started_at

    def eta(self, current: int, total: int) -> float | None:
        if current <= 0 or total <= 0 or current >= total:
            return 0.0 if total > 0 and current >= total else None
        elapsed = self.elapsed()
        return elapsed * (total - current) / current


class LineProgressReporter(ProgressReporter):
    def emit(self, event: ProgressEvent) -> None:
        label = self._label(event)
        parts = [label]
        for key in ("suite", "scenario", "controller", "estimator", "seed", "sample_id", "iteration", "item_name", "note"):
            value = event.context.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        if event.elapsed_s is not None:
            parts.append(f"elapsed={format_eta(event.elapsed_s)}")
        if event.eta_s is not None:
            parts.append(f"eta={format_eta(event.eta_s)}")
        sys.stderr.write(" | ".join(parts) + "\n")
        sys.stderr.flush()

    def _label(self, event: ProgressEvent) -> str:
        if event.current is not None and event.total is not None:
            percent = format_percent(event.current, event.total)
            if percent is not None:
                return f"[{event.domain} {percent} {event.current}/{event.total}] {event.stage}"
            return f"[{event.domain} {event.current}/{event.total}] {event.stage}"
        return f"[{event.domain}:{event.stage}]"
