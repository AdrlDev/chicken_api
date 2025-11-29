# app/ws_manager.py
import asyncio
from fastapi import WebSocket
from typing import List

class WebSocketManager:
    def __init__(self, buffer_size: int = 200):
        self.active: List[WebSocket] = []
        self.buffer: List[str] = []
        self.buffer_size = buffer_size
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        # synchronous because called from asyncio context only
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: str):
        """Send message to all connected clients (async)."""
        async with self._lock:
            # update buffer
            self.buffer.append(message)
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)

            # send to all clients
            to_remove = []
            for ws in list(self.active):
                try:
                    await ws.send_text(message)
                except Exception:
                    # mark for removal — can't await disconnect safely here
                    to_remove.append(ws)
            for r in to_remove:
                if r in self.active:
                    self.active.remove(r)

    async def send_buffer_to(self, ws: WebSocket):
        """When a client connects, send the buffered recent logs first."""
        for line in self.buffer:
            try:
                await ws.send_text(line)
            except Exception:
                break

# Singleton
ws_manager = WebSocketManager()
