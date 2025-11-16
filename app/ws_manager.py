# WebSocket manager

from fastapi import WebSocket

class WSManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def send(self, msg: str):
        for ws in self.connections:
            try:
                await ws.send_text(msg)
            except:
                self.disconnect(ws)

ws_manager = WSManager()