from __future__ import annotations

from contextvars import ContextVar


request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    request_id_var.set(request_id)


def get_request_id() -> str | None:
    return request_id_var.get()
