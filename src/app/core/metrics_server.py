from __future__ import annotations

from threading import Lock

from prometheus_client import start_http_server

_started_ports: set[int] = set()
_lock = Lock()


def start_metrics_server(port: int, address: str = "0.0.0.0") -> None:
    if port <= 0:
        return
    with _lock:
        if port in _started_ports:
            return
        start_http_server(port, addr=address)
        _started_ports.add(port)
