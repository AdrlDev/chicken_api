# main.py

import io
import os
import cv2
import base64
import threading
import numpy as np
from dotenv import load_dotenv
import asyncio
import time
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
from app.detection import _run_detection
from app.utils import yolo
from app.config import (
    IMAGES_DIR, 
    LABELS_DIR, 
    CONFIDENCE_THRESHOLD,
    WS_MAX_CONNECTIONS,
    DATASET_DIR
)
from app.process_image import process_image, processing_tasks, AutoLabelResponse

from app.ws_manager import ws_manager
from app.trainer_ws import start_training_thread

from contextlib import asynccontextmanager
from app.auth.login import router as auth_router
from app.auth.database import init_db
from app.chicken_scans.routes import router as scan_router
from app.chicken_scans.db import setup_db # Import the setup function

# Create necessary directories
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(LABELS_DIR).mkdir(parents=True, exist_ok=True)

MAX_QUEUE_SIZE = 1   # Always keep only the latest frame


train_script = os.path.join(os.path.dirname(__file__), "app", "train_model.py")

# Define response models
class DetectionResponse(BaseModel):
    label: str
    confidence: float
    bbox: List[float]
    timestampMs: int

class TrainResponse(BaseModel):
    status: str
    dataset: str

class TrainStatusResponse(BaseModel):
    session: str
    recent_logs: Optional[List[str]] = None
    status: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("Initializing database...")
    await init_db()
    await setup_db()
    print("Database initialized.")
    yield
    # shutdown (optional cleanup)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="ChickenAI",
    version="1.0",
    description="API for chicken disease detection and model training using YOLOv8",
    lifespan=lifespan
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

# POST endpoint to start training (non-blocking)
@app.post("/train-model")
async def start_training():
    """
    Start YOLO training in a separate background thread.
    Returns immediately.
    """
    start_training_thread(dataset_dir=str(DATASET_DIR), epochs=100, imgsz=640, val_ratio=0.2)
    return {"message": "Training started"}

# WebSocket endpoint for logs
@app.websocket("/ws/train")
async def websocket_train(ws: WebSocket):
    """
    WebSocket endpoint for streaming training logs.
    """
    await ws_manager.connect(ws)
    try:
        while True:
            await asyncio.sleep(1)  # keep connection alive
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)

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
                break
            except Exception as e:
                await websocket.send_json({"error": f"Failed to receive data: {str(e)}"})
                continue

            # Decode image
            try:
                image_bytes = base64.b64decode(data.split(",")[1])
                np_img = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Failed to decode image")
            except Exception as e:
                await websocket.send_json({"error": f"Invalid image data: {str(e)}"})
                continue

            # Run detection with proper input size
            try:
                results = yolo.predict(frame)

                detections = []
                for r in results:
                    for box in r.boxes: # type: ignore
                        conf = float(box.conf[0])
                        if conf < CONFIDENCE_THRESHOLD:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        label = yolo.names[cls]

                        # Draw box for visualization
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        detections.append(DetectionResponse(
                            label=label,
                            confidence=round(conf, 2),
                            bbox=[x1, y1, x2, y2],
                            timestampMs=int(datetime.now().timestamp() * 1000)
                        ))

                # Optional: encode annotated frame back to base64 to visualize in client
                _, buffer = cv2.imencode(".jpg", frame)
                annotated_b64 = base64.b64encode(buffer).decode("utf-8")

                await websocket.send_json({
                    "detections": [det.dict() for det in detections],
                    "annotated_image": f"data:image/jpeg;base64,{annotated_b64}"
                })

            except Exception as e:
                await websocket.send_json({"error": f"Detection error: {str(e)}"})

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
    Optimized real-time detection WebSocket:
    - Sequential receive
    - Async YOLO worker
    - Smooth boxes, low latency
    """
    await manager.connect(websocket)
    print("📡 Client connected for video stream")

    frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(MAX_QUEUE_SIZE)
    result_queue: asyncio.Queue[dict] = asyncio.Queue()
    stop_event = asyncio.Event()

    # ---------------------------
    # 👇 YOLO worker (process frames)
    # ---------------------------
    async def yolo_worker():
        last_time = time.time()
        while not stop_event.is_set():
            frame = await frame_queue.get()
            if frame is None:
                break

            # YOLO inference
            try:
                results = yolo(frame)  # type: ignore
                detections: list[dict] = []

                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf < CONFIDENCE_THRESHOLD:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        label = yolo.names[cls]

                        detections.append({
                            "label": label,
                            "confidence": round(conf, 2),
                            "bbox": [x1, y1, x2, y2],
                            "timestampMs": int(datetime.now().timestamp() * 1000)
                        })

                now = time.time()
                fps = 1 / (now - last_time)
                last_time = now

                await result_queue.put({
                    "detections": detections,
                    "fps": round(fps, 1)
                })

            except Exception as e:
                await result_queue.put({"error": f"Detection error: {str(e)}"})

    # Start YOLO worker
    worker_task = asyncio.create_task(yolo_worker())

    # ---------------------------
    # 👇 Frame receiving & sending loop
    # ---------------------------
    try:
        while True:
            # 1️⃣ Receive frame from client
            try:
                data = await websocket.receive_bytes()
            except WebSocketDisconnect:
                break
            except Exception as e:
                print("Receive error:", e)
                break

            # 2️⃣ Decode frame
            np_img = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 3️⃣ Put frame into queue (drop oldest if full)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except:
                    pass
            await frame_queue.put(frame)

            # 4️⃣ Send all available results
            while not result_queue.empty():
                result = await result_queue.get()
                await websocket.send_json(result)

    finally:
        stop_event.set()
        worker_task.cancel()
        manager.disconnect(websocket)
        print("✔ WebSocket cleaned up")

# ---------------------------------
# 🐔 AUTHENTICATION ENDPOINTS
# ---------------------------------

app.include_router(auth_router)
# Include the modularized router
app.include_router(scan_router)