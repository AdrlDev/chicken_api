from fastapi import FastAPI, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse
import os
import cv2
from ultralytics import YOLO # type: ignore
from app.train_model import train_yolo_autosplit  # ✅ import from app folder
import threading
import base64
import numpy as np

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
YOLO_WEIGHTS = os.path.join(BASE_DIR, "assets", "yolov8n.pt")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINED_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")

# --- Load initial model ---
yolo = YOLO(TRAINED_WEIGHTS if os.path.exists(TRAINED_WEIGHTS) else YOLO_WEIGHTS)

app = FastAPI(title="ChickenAI", version="1.0")

# --- Global flag to stop live detection ---
stop_live = False

@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    """
    Trigger YOLO training using dataset from Label Studio.
    Runs in background so the API stays responsive.
    """
    def _train():
        print("🚀 Starting YOLO training...")
        train_yolo_autosplit(
            dataset_dir=DATASET_DIR,
            model_name=YOLO_WEIGHTS,
            epochs=100,
            imgsz=640,
            val_ratio=0.2
        )

        # Reload trained weights automatically
        new_weights = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
        if os.path.exists(new_weights):
            global yolo
            yolo = YOLO(new_weights)
            print(f"✅ Model reloaded with new weights: {new_weights}")

    background_tasks.add_task(_train)
    return JSONResponse({
        "status": "training started",
        "dataset": DATASET_DIR
    })

@app.get("/detect/live")
def detect_live():
    """
    Start live detection from webcam using trained YOLO model.
    Press 'q' to stop the window.
    """
    def _run_detection():
        global stop_live
        stop_live = False
        print("🎥 Starting live chicken detection...")

        cap = cv2.VideoCapture(0)  # webcam index
        if not cap.isOpened():
            print("❌ Cannot open camera.")
            return

        while not stop_live:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No frame captured.")
                break

            # Run YOLO inference
            results = yolo(frame, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = yolo.names[cls]

                    # Draw bounding box
                    color = (0, 255, 0) if label.lower() == "healthy" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("ChickenAI Live Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Stopping live detection...")
                break

        cap.release()
        cv2.destroyAllWindows()
        stop_live = True
        print("✅ Live detection ended.")

    threading.Thread(target=_run_detection).start()
    return JSONResponse({"status": "live detection started"})


@app.post("/detect/stop")
def stop_detection():
    """
    Stop the live detection loop.
    """
    global stop_live
    stop_live = True
    return JSONResponse({"status": "live detection stopped"})


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    print("📡 Client connected to realtime YOLO")

    try:
        while True:
            data = await websocket.receive_text()

            # Decode base64 image
            image_bytes = base64.b64decode(data.split(",")[1])
            np_img = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # Run YOLO detection
            results = yolo(frame)
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = yolo.names[cls]
                    detections.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2]
                    })

            await websocket.send_json({"detections": detections})
    except Exception as e:
        print("❌ Connection closed:", e)
    finally:
        await websocket.close()
        print("🛑 WebSocket disconnected")