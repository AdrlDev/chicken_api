# main.py

import io
import os
import cv2
import json
import base64
import threading
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from app.label_studio import label_studio  # Import our token manager
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

# Dictionary to store processing status
processing_tasks: Dict[str, Dict] = {}

# ---------------------------------
# 🐔 AUTO-LABEL ENDPOINT (ASYNC)
# ---------------------------------
@app.post("/auto-label-train", response_model=UploadResponse)
async def auto_label_train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    label_name: str = Form(...)  # Make label_name required
):
    """
    Upload an image with a required label for training.
    
    Every image must have a label specified (e.g., 'healthy', 'fowl-pox', etc.).
    The function will:
    1. Detect chicken objects in the image
    2. Label all detected chickens with the provided label
    3. Save the image and its annotations
    4. Update the training dataset
    
    Args:
        file: Image file to process (must be JPEG or PNG)
        label_name: Required label name (e.g., 'healthy', 'fowl-pox', etc.)
    
    Returns:
        UploadResponse: Initial response with task ID
    """
    # Validate label name
    if not label_name or not label_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Label name is required"
        )
    # Generate unique task ID and filename
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    image_filename = f"auto_{task_id}.jpg"
    image_path = str(Path(IMAGES_DIR) / image_filename)
    
    try:

        # Initial validation of the image
        file_contents = await file.read()
        try:
            img = Image.open(io.BytesIO(file_contents))
            img.verify()
            if img.format not in ['JPEG', 'PNG']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only JPEG and PNG images are supported"
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image: {str(e)}"
            )

        # Save the image
        with open(image_path, "wb") as f:
            f.write(file_contents)

        # Store task information
        processing_tasks[task_id] = {
            "status": "processing",
            "image_path": image_path,
            "label_name": label_name
        }

        # Start processing in background
        background_tasks.add_task(
            process_image,
            task_id=task_id,
            image_path=image_path,
            label_name=label_name
        )

        return UploadResponse(
            message="Image uploaded successfully, processing started",
            image_id=task_id
        )

    except HTTPException as e:
        if os.path.exists(image_path):
            os.unlink(image_path)
        raise e
    except Exception as e:
        if os.path.exists(image_path):
            os.unlink(image_path)
        raise HTTPException(status_code=500, detail=str(e))

async def process_image(task_id: str, image_path: str, label_name: str):
    """
    Process an uploaded image, detect chickens, and label them.
    
    Args:
        task_id: Unique identifier for this processing task
        image_path: Path to the uploaded image
        label_name: The label to apply to all detected chickens
    """
    try:
        # Get fresh Label Studio client with updated token
        ls_client = label_studio.get_client()
        
        # Load existing classes
        class_names = []
        if CLASSES_PATH.exists():
            with open(CLASSES_PATH, "r") as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]

        # Add new label to classes if not present
        if label_name not in class_names:
            class_names.append(label_name)
            # Save updated classes immediately
            with open(CLASSES_PATH, "w") as f:
                f.write("\n".join(class_names))

        # Get class index for the label
        label_index = class_names.index(label_name)
        
        detections = []

        # Detect chicken objects in the image
        results = yolo.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, save=False) # type: ignore
        if not results or len(results[0].boxes) == 0:  # type: ignore
            processing_tasks[task_id]["status"] = "error"
            processing_tasks[task_id]["error"] = "No chicken objects detected in the image"
            return

        # Process each detected chicken
        for box in results[0].boxes:  # type: ignore
            # Get normalized bounding box coordinates (YOLO format)
            x_center, y_center, width, height = box.xywhn[0].tolist()
            
            # Create detection with the provided label
            detections.append(DetectionResponse(
                label=label_name,
                confidence=float(box.conf[0]),
                bbox=[x_center, y_center, width, height]
            ))

        # Save YOLO label file
        image_filename = Path(image_path).name
        label_filename = Path(image_path).stem + ".txt"
        label_path = LABELS_DIR / label_filename
        
        with open(label_path, "w") as f:
            for det in detections:
                # Use the label_index determined earlier for all detections
                f.write(f"{label_index} {det.bbox[0]:.6f} {det.bbox[1]:.6f} {det.bbox[2]:.6f} {det.bbox[3]:.6f}\n")

        # Move image to dataset/images
        dataset_img_path = Path(DATASET_DIR) / "images" / image_filename
        Path(dataset_img_path).parent.mkdir(parents=True, exist_ok=True)
        os.rename(image_path, dataset_img_path)

        # Update notes.json with metadata
        notes_path = Path(DATASET_DIR) / "notes.json"
        notes = {}
        if notes_path.exists():
            with open(notes_path, "r") as f:
                notes = json.load(f)

        notes[image_filename] = {
            "label": label_name,
            "upload_date": datetime.now().isoformat(),
            "detections": len(detections)
        }

        with open(notes_path, "w") as f:
            json.dump(notes, f, indent=4)

        # Try to sync with Label Studio
        try:
            # Verify connection is still valid
            projects = ls_client.list_projects()
            print(f"Label Studio connection verified, found {len(projects)} projects") # type: ignore
        except Exception as ls_err:
            print(f"Label Studio sync error (non-critical): {str(ls_err)}")

        # Store results
        processing_tasks[task_id].update({
            "status": "completed",
            "result": AutoLabelResponse(
                message="✅ Image labeled successfully",
                mode="manual",
                image=str(dataset_img_path),
                label_file=str(label_path),
                label_name=label_name,
                classes=class_names
            )
        })

        # Trigger training in background
        _train_auto(epochs=AUTO_TRAIN_EPOCHS, imgsz=AUTO_TRAIN_IMAGE_SIZE)

    except Exception as e:
        processing_tasks[task_id].update({
            "status": "error",
            "error": str(e)
        })

@app.get("/auto-label-train/{task_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(task_id: str):
    """
    Get the status of an image processing task
    
    Args:
        task_id: The ID of the processing task
        
    Returns:
        ProcessingStatusResponse: Current status of the task
    """
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
        
    task = processing_tasks[task_id]
    return ProcessingStatusResponse(
        status=task["status"],
        result=task.get("result"),
        error=task.get("error")
    )

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
