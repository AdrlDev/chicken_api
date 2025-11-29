# app/routes/detection_ws.py

import asyncio
import base64
import time
import cv2
import numpy as np
from datetime import datetime
from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from app.utils.utils import yolo
from app.utils.config import CONFIDENCE_THRESHOLD, WS_MAX_CONNECTIONS
from app.main import ConnectionManager # Need to import the manager
from app.utils.ws_manager import ws_manager as train_ws_manager # Use alias for clarity

router = APIRouter(
    tags=["Realtime Detection"],
)

# Define detection response model (moved here)
class DetectionResponse(BaseModel):
    label: str
    confidence: float
    bbox: List[float]
    timestampMs: int

# Use the same manager as defined in main.py
manager = ConnectionManager()
MAX_QUEUE_SIZE = 1 

# WebSocket endpoint for training logs (moved here)
@router.websocket("/ws/train")
async def websocket_train(ws: WebSocket):
    """WebSocket endpoint for streaming training logs."""
    await train_ws_manager.connect(ws)
    try:
        while True:
            await asyncio.sleep(1)  # keep connection alive
    except WebSocketDisconnect:
        train_ws_manager.disconnect(ws)

@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint for real-time object detection from camera feed."""
    try:
        await manager.connect(websocket)
        print(f"📡 WebSocket client connected ({manager.connection_count}/{WS_MAX_CONNECTIONS})")

        while True:
            # ... (rest of /ws/detect logic remains the same) ...
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
        print(f" Active connections: {manager.connection_count}/{WS_MAX_CONNECTIONS}")


@router.websocket("/ws/video-detect")
async def websocket_video_detect(websocket: WebSocket):
    """Optimized real-time detection WebSocket for video file processing."""
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
                        # ... (rest of detection logic) ...
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