from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class StageMetrics:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, duration_ms: float) -> None:
        self.count += 1
        self.total_ms += duration_ms
        self.max_ms = max(self.max_ms, duration_ms)

    @property
    def average_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return round(self.total_ms / self.count, 2)


@dataclass
class QueryMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    total_latency_ms: float = 0.0
    confidence_total: float = 0.0
    citations_total: int = 0
    stage_metrics: dict[str, StageMetrics] = field(default_factory=dict)
    events: dict[str, int] = field(default_factory=dict)


class MetricsRegistry:
    def __init__(self) -> None:
        self._metrics = QueryMetrics()
        self._lock = Lock()

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        with self._lock:
            stage = self._metrics.stage_metrics.setdefault(stage_name, StageMetrics())
            stage.record(duration_ms)

    def record_event(self, event_name: str) -> None:
        with self._lock:
            self._metrics.events[event_name] = self._metrics.events.get(event_name, 0) + 1

    def record_success(
        self,
        total_latency_ms: float,
        *,
        confidence_score: float,
        citation_count: int,
        cache_hit: bool,
    ) -> None:
        with self._lock:
            self._metrics.total_requests += 1
            self._metrics.successful_requests += 1
            self._metrics.total_latency_ms += total_latency_ms
            self._metrics.confidence_total += confidence_score
            self._metrics.citations_total += citation_count
            if cache_hit:
                self._metrics.cache_hits += 1

    def record_failure(self, total_latency_ms: float, *, cache_hit: bool = False) -> None:
        with self._lock:
            self._metrics.total_requests += 1
            self._metrics.failed_requests += 1
            self._metrics.total_latency_ms += total_latency_ms
            if cache_hit:
                self._metrics.cache_hits += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total_requests = self._metrics.total_requests
            successful_requests = self._metrics.successful_requests
            average_latency_ms = round(
                self._metrics.total_latency_ms / total_requests, 2
            ) if total_requests else 0.0
            average_confidence = round(
                self._metrics.confidence_total / successful_requests, 4
            ) if successful_requests else 0.0
            average_citations = round(
                self._metrics.citations_total / successful_requests, 2
            ) if successful_requests else 0.0
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": self._metrics.failed_requests,
                "cache_hits": self._metrics.cache_hits,
                "average_latency_ms": average_latency_ms,
                "average_confidence": average_confidence,
                "average_citations": average_citations,
                "events": dict(self._metrics.events),
                "stage_metrics": {
                    name: {
                        "count": metrics.count,
                        "average_ms": metrics.average_ms,
                        "max_ms": round(metrics.max_ms, 2),
                    }
                    for name, metrics in self._metrics.stage_metrics.items()
                },
            }
