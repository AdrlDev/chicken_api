# main.py

import io
import os
import cv2
import base64
import threading
import numpy as np
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from fastapi import (
    FastAPI, 
    BackgroundTasks, 
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

from app.train_model import _train, _train_auto
from app.detection import _run_detection
from app.utils import yolo
from app.config import (
    DATASET_DIR, 
    IMAGES_DIR, 
    LABELS_DIR, 
    BASE_DIR,
    CLASSES_PATH,
    LOGS_DIR,
    CONFIDENCE_THRESHOLD,
    WEBSOCKET_CONFIDENCE_THRESHOLD,
    AUTO_TRAIN_EPOCHS,
    AUTO_TRAIN_IMAGE_SIZE,
    WS_MAX_CONNECTIONS
)

# Create necessary directories
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(LABELS_DIR).mkdir(parents=True, exist_ok=True)

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

# ---------------------------------
# 🧩 TRAIN MODEL (with Background Task)
# ---------------------------------
@app.post("/train", response_model=TrainResponse)
async def train_model(background_tasks: BackgroundTasks):
    """
    Trigger YOLO training using dataset from Label Studio.
    
    Returns:
        TrainResponse: Status of the training process
    """
    try:
        background_tasks.add_task(_train)
        return TrainResponse(
            status="training started",
            dataset=str(DATASET_DIR)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )

# Response models for auto-label
class AutoLabelResponse(BaseModel):
    message: str
    mode: str
    image: str
    label_file: str
    label_name: str
    classes: List[str]

# ---------------------------------
# 🐔 AUTO-LABEL ENDPOINT (ASYNC)
# ---------------------------------
@app.post("/auto-label-train", response_model=AutoLabelResponse)
async def auto_label_train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    label_name: str = Form(None)
):
    """
    Upload an image and optionally provide a label for training.
    
    The endpoint supports two modes:
    1. Auto-labeling: Model predicts labels automatically
    2. Manual labeling: User provides a specific label
    
    Args:
        file: Image file to process
        label_name: Optional manual label name
    
    Returns:
        AutoLabelResponse: Details about the processed image and training status
    """
    try:
        # Generate unique filename
        image_filename = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        image_path = str(IMAGES_DIR / image_filename)

        # Validate and save image
        file_contents = await file.read()
        try:
            img = Image.open(io.BytesIO(file_contents))
            img.verify()  # verify corruption
            
            # Additional image validation
            if img.format not in ['JPEG', 'PNG']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only JPEG and PNG images are supported"
                )
        except UnidentifiedImageError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image validation failed: {str(e)}"
            )

        # Save the validated image
        with open(image_path, "wb") as f:
            f.write(file_contents)

        # Load existing classes
        class_names = []
        if CLASSES_PATH.exists():
            with open(CLASSES_PATH, "r") as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]

        detections = []

        # Manual labeling mode
        if label_name:
            if label_name not in class_names:
                class_names.append(label_name)
            detections.append(DetectionResponse(
                label=label_name,
                confidence=1.0,
                bbox=[0.5, 0.5, 1.0, 1.0]  # full image placeholder
            ))
        else:
            # Auto-label mode
            results = yolo.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, save=False)
            if not results or len(results[0].boxes) == 0: # type: ignore
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No objects detected in the image with sufficient confidence"
                )
            
            for box in results[0].boxes: # type: ignore
                cls_id = int(box.cls[0].item())
                auto_label = yolo.names[cls_id]
                if auto_label not in class_names:
                    class_names.append(auto_label)
                x_center, y_center, width, height = box.xywhn[0].tolist()
                detections.append(DetectionResponse(
                    label=auto_label,
                    confidence=float(box.conf[0]),
                    bbox=[x_center, y_center, width, height]
                ))

        # Save YOLO label file
        label_filename = image_filename.replace(".jpg", ".txt")
        label_path = str(LABELS_DIR / label_filename)
        with open(label_path, "w") as f:
            for det in detections:
                label_index = class_names.index(det.label)
                x_center, y_center, width, height = det.bbox
                f.write(f"{label_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Save updated classes
        with open(CLASSES_PATH, "w") as f:
            f.write("\n".join(class_names))

        # Trigger incremental auto-training
        background_tasks.add_task(_train_auto, epochs=AUTO_TRAIN_EPOCHS, imgsz=AUTO_TRAIN_IMAGE_SIZE)

        return AutoLabelResponse(
            message="✅ Image validated, labeled, and incremental fine-tuning started",
            mode="manual" if label_name else "auto",
            image=image_path,
            label_file=label_path,
            label_name=label_name or "auto-detected",
            classes=class_names
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
                results = yolo(frame)
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
                        label = yolo.names[cls]
                        
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
                results = yolo(frame)
                detections: List[DetectionResponse] = []

                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf < CONFIDENCE_THRESHOLD:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        label = yolo.names[cls]
                        
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
