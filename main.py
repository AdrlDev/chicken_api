# main.py

import io
import os
import cv2
import base64
import threading
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image
from fastapi import (
    FastAPI, 
    WebSocket, 
    File, 
    UploadFile, 
    HTTPException, 
    Form,
    status,
    WebSocketDisconnect
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import asyncio

from app.train_model import _train
from app.detection import _run_detection
from app.utils import yolo
from app.config import (
    DATASET_DIR, 
    IMAGES_DIR, 
    LABELS_DIR, 
    LOGS_DIR,
    CONFIDENCE_THRESHOLD,
    WEBSOCKET_CONFIDENCE_THRESHOLD,
    WS_MAX_CONNECTIONS
)
from app.process_image import process_image, processing_tasks, AutoLabelResponse
import subprocess
import sys

# Create necessary directories
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(LABELS_DIR).mkdir(parents=True, exist_ok=True)

train_script = os.path.join(os.path.dirname(__file__), "app", "train_model.py")

# Define response models
class DetectionResponse(BaseModel):
    label: str
    confidence: float
    bbox: List[float]

class TrainResponse(BaseModel):
    status: str
    dataset: str

class TrainStatusResponse(BaseModel):
    session: str
    recent_logs: Optional[List[str]] = None
    status: Optional[str] = None

# Initialize FastAPI app with metadata
app = FastAPI(
    title="ChickenAI",
    version="1.0",
    description="API for chicken disease detection and model training using YOLOv8"
)

stop_live = False  # global flag for webcam detection

# Response model for initial upload response
class UploadResponse(BaseModel):
    message: str
    image_id: str
    status: str = "processing"

# Response model for checking processing status
class ProcessingStatusResponse(BaseModel):
    status: str
    result: Optional[AutoLabelResponse] = None
    error: Optional[str] = None

LS_URL = os.getenv("LABEL_STUDIO_URL")
LS_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

# Serve dataset/images at /dataset/images
images_dir = os.path.join(os.getcwd(), "dataset/images")
app.mount("/dataset/images", StaticFiles(directory=images_dir), name="images")

# ---------------------------------
# 🐔 AUTO-LABEL ENDPOINT (ASYNC)
# ---------------------------------
@app.post("/auto-label-train", response_model=UploadResponse)
async def auto_label_train(
    file: UploadFile = File(...),
    label_name: str = Form(...)  # Make label_name required
):
    """
    Upload an image with a label (e.g., 'healthy', 'fowl-pox') and auto-label detected chickens.
    """
    # Validate label name
    if not label_name or not label_name.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Label name is required")

    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    image_filename = f"auto_{task_id}.jpg"
    image_path = Path(IMAGES_DIR) / image_filename

    try:
        # Validate image
        file_contents = await file.read()
        try:
            img = Image.open(io.BytesIO(file_contents))
            img.verify()
            if img.format not in ["JPEG", "PNG"]:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only JPEG and PNG supported")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {str(e)}")

        # Save image to local storage
        with open(image_path, "wb") as f:
            f.write(file_contents)

        # Add task to processing_tasks dictionary
        processing_tasks[task_id] = {
            "status": "processing",
            "image_path": str(image_path),
            "label_name": label_name
        }

        # Start background task for processing
        asyncio.create_task(process_image(
            task_id=task_id,
            image_path=str(image_path),
            label_name=label_name
        ))

        return UploadResponse(
            message="Image uploaded successfully, processing started",
            image_id=task_id
        )

    except Exception as e:
        if image_path.exists():
            os.unlink(image_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auto-label-train/{task_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(task_id: str):
    """
    Get the status of an image processing task.
    """
    if task_id not in processing_tasks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    task = processing_tasks[task_id]
    return ProcessingStatusResponse(
        status=task["status"],
        result=task.get("result"),
        error=task.get("error")
    )

@app.post("/train-model")
async def train_model():
    # Launch training in background subprocess
    subprocess.Popen(
        [sys.executable, "-u", train_script],  # adjust path if needed
        stdout=sys.stdout,  # <-- prints to VPS logs
        stderr=sys.stderr,  # <-- prints errors to VPS logs
        bufsize=1,
        universal_newlines=True
    )
    return {"message": "Training started in background"}

# ---------------------------------
# 🎥 LIVE DETECTION (Webcam)
# ---------------------------------
@app.get("/detect/live")
def detect_live():
    """Start live detection from webcam using YOLO model."""
    threading.Thread(target=_run_detection).start()
    return JSONResponse({"status": "live detection started"})

@app.post("/detect/stop")
def stop_detection():
    """Stop the live detection loop."""
    global stop_live
    stop_live = True
    return JSONResponse({"status": "live detection stopped"})

@app.get("/train/status", response_model=TrainStatusResponse)
async def train_status():
    """
    Get the status of the latest training session.
    
    Returns:
        TrainStatusResponse: Information about the latest training session
    """
    try:
        if not LOGS_DIR.exists():
            return TrainStatusResponse(
                session="none",
                status="No training sessions found"
            )

        sessions = [d for d in os.listdir(LOGS_DIR) if d.startswith("train_")]
        if not sessions:
            return TrainStatusResponse(
                session="none",
                status="No training sessions yet"
            )

        latest = sorted(sessions)[-1]
        log_file = LOGS_DIR / latest / "results.txt"
        
        if not log_file.exists():
            return TrainStatusResponse(
                session=latest,
                status="Training in progress - no logs yet"
            )

        with open(log_file, "r") as f:
            lines = f.readlines()[-10:]  # Get last 10 lines
            
        return TrainStatusResponse(
            session=latest,
            recent_logs=lines
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}"
        )


# ---------------------------------
# 🌐 REALTIME DETECTION (WebSocket)
# ---------------------------------
# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        if len(self.active_connections) >= WS_MAX_CONNECTIONS:
            raise WebSocketDisconnect(code=1008, reason="Maximum connections reached")
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    @property
    def connection_count(self):
        return len(self.active_connections)

manager = ConnectionManager()

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time object detection from camera feed.
    
    Expects: Base64 encoded image data
    Returns: JSON with detection results
    """
    try:
        await manager.connect(websocket)
        print(f"📡 WebSocket client connected ({manager.connection_count}/{WS_MAX_CONNECTIONS})")

        while True:
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                raise  # Re-raise to handle in outer try/except
            except Exception as e:
                await websocket.send_json({
                    "error": f"Failed to receive data: {str(e)}"
                })
                continue
                
            # Decode and process image
            try:
                image_bytes = base64.b64decode(data.split(",")[1])
                np_img = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                
                if frame is None:
                    raise ValueError("Failed to decode image")
                    
                h, w, _ = frame.shape
            except Exception as e:
                await websocket.send_json({
                    "error": f"Invalid image data: {str(e)}"
                })
                continue

            # Run detection
            try:
                results = yolo(frame) # type: ignore
                detections: List[DetectionResponse] = []

                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf < WEBSOCKET_CONFIDENCE_THRESHOLD:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Filter out detections that cover most of the image
                        if (x2 - x1) > 0.9 * w and (y2 - y1) > 0.9 * h:
                            continue
                            
                        cls = int(box.cls[0])
                        label = yolo.names[cls] # type: ignore
                        
                        detections.append(DetectionResponse(
                            label=label,
                            confidence=round(conf, 2),
                            bbox=[x1, y1, x2, y2]
                        ))

                await websocket.send_json({"detections": [det.dict() for det in detections]})
            except Exception as e:
                await websocket.send_json({
                    "error": f"Detection error: {str(e)}"
                })

    except WebSocketDisconnect:
        print(f"🛑 Client disconnected normally")
    except Exception as e:
        print(f"❌ WebSocket error: {str(e)}")
    finally:
        manager.disconnect(websocket)
        print(f"� Active connections: {manager.connection_count}/{WS_MAX_CONNECTIONS}")

# ---------------------------------
# 🎬 WEBSOCKET: VIDEO FILE LIVE DETECTION
# ---------------------------------
@app.websocket("/ws/video-detect")
async def websocket_video_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time object detection from video stream.
    
    Expects: Binary JPEG frames
    Returns: JSON with detection results
    """
    try:
        await manager.connect(websocket)
        print(f"📡 Client connected for video stream ({manager.connection_count}/{WS_MAX_CONNECTIONS})")

        while True:
            try:
                # Receive binary frame (JPEG bytes)
                frame_bytes = await websocket.receive_bytes()

                # Decode image
                try:
                    np_img = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Failed to decode frame")
                except Exception as e:
                    await websocket.send_json({
                        "error": f"Invalid frame data: {str(e)}"
                    })
                    continue

                # Run YOLO detection
                results = yolo(frame) # type: ignore
                detections: List[DetectionResponse] = []

                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf < CONFIDENCE_THRESHOLD:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        label = yolo.names[cls] # type: ignore
                        
                        detections.append(DetectionResponse(
                            label=label,
                            confidence=round(conf, 2),
                            bbox=[x1, y1, x2, y2]
                        ))

                # Send results back to client
                await websocket.send_json([det.dict() for det in detections])
                
            except WebSocketDisconnect:
                raise  # Re-raise to handle in outer try/except
            except Exception as e:
                await websocket.send_json({
                    "error": f"Processing error: {str(e)}"
                })

    except WebSocketDisconnect:
        print(f"🛑 Video stream client disconnected normally")
    except Exception as e:
        print(f"❌ Video stream error: {str(e)}")
    finally:
        manager.disconnect(websocket)
        print(f"� Active video streams: {manager.connection_count}/{WS_MAX_CONNECTIONS}")
