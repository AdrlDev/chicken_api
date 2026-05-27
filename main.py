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
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Create necessary directories
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(LABELS_DIR).mkdir(parents=True, exist_ok=True)

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

# ----------------------------------------------------------------
# ⭐️ FIX: Add CORSMiddleware
# ----------------------------------------------------------------

origins = [
    # ⭐️ ALLOW YOUR FRONTEND ORIGIN (localhost)
     "http://localhost:3000",
    "http://localhost:3001",   # ← add this
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",   # ← and this
    
    # ⭐️ ALLOW YOUR DEPLOYED FRONTEND ORIGIN (If different from backend)
    # E.g., if the frontend is served from https://chickens.com
    # "https://chickens.com", 
    
    # If the backend and frontend are the same origin, you don't need this, 
    # but for local dev, you definitely do.
    
    # NOTE: You can use ["*"] for development, but it's less secure.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,                      # Specify the allowed origins
    allow_credentials=True,                     # Allow cookies/auth headers
    allow_methods=["*"],                        # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*", "Authorization"],       # Allow all headers, including custom ones like Authorization
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