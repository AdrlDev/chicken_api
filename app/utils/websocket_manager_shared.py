# app/utils/websocket_manager_shared.py

from typing import List
from fastapi import WebSocket, WebSocketDisconnect
from app.utils.config import WS_MAX_CONNECTIONS # Re-use your config constant

# ---------------------------------
# 🔴 GLOBAL FLAG FOR LIVE DETECTION
# ---------------------------------
stop_live = False  # The global flag for webcam detection

# ---------------------------------
# 🔵 WEBSOCKET CONNECTION MANAGER
# ---------------------------------
class ConnectionManager:
    """Manager for realtime detection WebSockets."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        if len(self.active_connections) >= WS_MAX_CONNECTIONS:
            # Must re-raise the specific exception
            raise WebSocketDisconnect(code=1008, reason="Maximum connections reached")
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    @property
    def connection_count(self):
        return len(self.active_connections)

# Create the singleton manager instance here
manager = ConnectionManager()