from __future__ import annotations

from uuid import UUID


def _stable_ratio(identifier: UUID) -> float:
    return identifier.int / float(1 << 128)


def should_route_report_to_ml(report_id: UUID, route_rate: float) -> bool:
    route_rate = max(0.0, min(1.0, route_rate))
    return _stable_ratio(report_id) < route_rate


def should_sample_manual_review(report_id: UUID, sample_rate: float) -> bool:
    sample_rate = max(0.0, min(1.0, sample_rate))
    return _stable_ratio(report_id) < sample_rate
