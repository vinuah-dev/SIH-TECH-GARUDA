"""WebSocket fan-out for live alerts.

The surveillance pipeline runs in a worker thread and knows nothing about the
web layer; it just calls its event hooks. This hub is the bridge: it takes a
call from that thread and hands it to the asyncio loop that owns the sockets.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlertHub:
    """Tracks connected dashboards and broadcasts events to them."""

    loop: asyncio.AbstractEventLoop | None = field(default=None, init=False)
    _clients: set[Any] = field(default_factory=set, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the loop that owns the sockets, at startup."""
        self.loop = loop

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        with self._lock:
            self._clients.add(websocket)

    def disconnect(self, websocket: Any) -> None:
        with self._lock:
            self._clients.discard(websocket)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    # ------------------------------------------------------------ publishing

    def publish_threadsafe(self, message: dict) -> None:
        """Called from the pipeline thread. Never raises into the pipeline."""
        if self.loop is None or self.loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)
        except RuntimeError:
            # The loop is shutting down; dropping a broadcast is acceptable,
            # the event is already in the store either way.
            pass

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(message, ensure_ascii=False, default=str)
        with self._lock:
            clients = list(self._clients)
        dead = []
        for client in clients:
            try:
                await client.send_text(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self.disconnect(client)
