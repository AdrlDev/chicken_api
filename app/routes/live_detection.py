# app/routes/live_detection.py

import threading
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.detection import _run_detection
from app.main import stop_live # Need to import the global flag

router = APIRouter(
    tags=["Live Detection"],
)

# ---------------------------------
# 🎥 LIVE DETECTION (Webcam)
# ---------------------------------
@router.get("/detect/live")
def detect_live():
    """Start live detection from webcam using YOLO model."""
    threading.Thread(target=_run_detection).start()
    return JSONResponse({"status": "live detection started"})

@router.post("/detect/stop")
def stop_detection():
    """Stop the live detection loop."""
    # NOTE: This relies on the global flag 'stop_live' defined in main.py
    global stop_live
    stop_live = True
    return JSONResponse({"status": "live detection stopped"})