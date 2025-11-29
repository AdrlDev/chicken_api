# main.py (SIMPLIFIED)

import os
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from app.utils.config import IMAGES_DIR, LABELS_DIR, WS_MAX_CONNECTIONS
from app.database.database import init_db
from app.auth.login import router as auth_router
from app.chicken_scans.routes import router as scan_router
# ⭐️ IMPORT NEW ROUTERS
from app.routes.train_and_data import router as train_router
from app.routes.detection_ws import router as websocket_router
from app.routes.live_detection import router as live_detect_router

from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables
load_dotenv()

# Create necessary directories
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(LABELS_DIR).mkdir(parents=True, exist_ok=True)

# ---------------------------------
# 🔴 KEEP: Global Variables/Managers
# ---------------------------------
stop_live = False  # global flag for webcam detection (used by live_detection.py)

class ConnectionManager:
    """Manager for realtime detection WebSockets."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        if len(self.active_connections) >= WS_MAX_CONNECTIONS:
            # Note: We must re-import WebSocketDisconnect here if we keep this class modular.
            from fastapi import WebSocketDisconnect 
            raise WebSocketDisconnect(code=1008, reason="Maximum connections reached")
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    @property
    def connection_count(self):
        return len(self.active_connections)

manager = ConnectionManager() # Used by detection_ws.py

# ---------------------------------
# 🚀 APP LIFESPAN & INITIALIZATION
# ---------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("Initializing database...")
    await init_db()
    print("Database initialized.")
    yield
    # shutdown (optional cleanup)

# Initialize FastAPI app
app = FastAPI(
    title="ChickenAI",
    version="1.0",
    description="API for chicken disease detection and model training using YOLOv8",
    lifespan=lifespan
)

# Serve dataset/images at /dataset/images
images_dir = os.path.join(os.getcwd(), "dataset/images")
app.mount("/dataset/images", StaticFiles(directory=images_dir), name="images")

# ---------------------------------
# 🧩 INCLUDE MODULAR ENDPOINTS
# ---------------------------------
app.include_router(auth_router)        # /login, /register
app.include_router(scan_router)         # /scans endpoints (assumed)
app.include_router(train_router)        # /auto-label-train, /train-model
app.include_router(websocket_router)    # /ws/detect, /ws/video-detect, /ws/train
app.include_router(live_detect_router)  # /detect/live, /detect/stop

# NOTE: The body of the original WebSocket functions (e.g., websocket_detect) 
# is now entirely contained within app/routes/detection_ws.py.