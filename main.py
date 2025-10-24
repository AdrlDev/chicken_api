#main.py

from fastapi import FastAPI, BackgroundTasks, WebSocket, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import cv2
from app.train_model import _train, _train_auto  # ✅ Your existing training utility
import threading
import base64
import numpy as np
import shutil
from pathlib import Path
from app.utils import DATASET_DIR, IMAGES_DIR, LABELS_DIR, BASE_DIR, yolo, timestamp, image_filename, image_path, classes_path
from app.detection import _run_detection

Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(LABELS_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ChickenAI", version="1.0")

stop_live = False  # global flag for webcam detection

# ---------------------------------
# 🧩 TRAIN MODEL (with Background Task)
# ---------------------------------
@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    """Trigger YOLO training using dataset from Label Studio."""
    background_tasks.add_task(_train)
    return JSONResponse({
        "status": "training started",
        "dataset": DATASET_DIR
    })


# ---------------------------------
# 🐔 AUTO-LABEL ENDPOINT (ASYNC)
# ---------------------------------
@app.post("/auto-label-train")
async def auto_label_train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    label_name: str = Form(None)
):
    """
    Upload an image + optional label → auto-label with YOLO or manual label → 
    update classes.txt → save YOLO label → trigger fine-tuning.
    """
    try:
        # ✅ 1. Save uploaded image
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ 2. Load existing classes
        classes_path = os.path.join(DATASET_DIR, "classes.txt")
        if os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            class_names = []

        detections = []

        if label_name:
            # ✅ Manual labeling mode
            print(f"📌 Manual label provided: {label_name}")
            # Add to class list if new
            if label_name not in class_names:
                class_names.append(label_name)
                print(f"🆕 New class added manually: {label_name}")

            detections.append({
                "label": label_name,
                "confidence": 1.0,
                "bbox": [0.5, 0.5, 1.0, 1.0]  # full image placeholder
            })

        else:
            # ✅ Auto-label mode using YOLO detection
            results = yolo.predict(source=image_path, conf=0.4, save=False)
            if not results or len(results[0].boxes) == 0:  # type: ignore
                raise HTTPException(status_code=400, detail="No detections found in image.")

            for box in results[0].boxes:  # type: ignore
                cls_id = int(box.cls[0].item())
                auto_label = yolo.names[cls_id]

                if auto_label not in class_names:
                    class_names.append(auto_label)
                    print(f"🆕 New class added: {auto_label}")

                x_center, y_center, width, height = box.xywhn[0].tolist()
                detections.append({
                    "label": auto_label,
                    "confidence": float(box.conf[0]),
                    "bbox": [x_center, y_center, width, height]
                })

        # ✅ 3. Save YOLO label file
        label_filename = image_filename.replace(".jpg", ".txt")
        label_path = os.path.join(LABELS_DIR, label_filename)

        with open(label_path, "w") as f:
            for det in detections:
                label_index = class_names.index(det["label"])
                x_center, y_center, width, height = det["bbox"]
                f.write(f"{label_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # ✅ 4. Save updated class list
        with open(classes_path, "w") as f:
            f.write("\n".join(class_names))

        # ✅ 5. Trigger background fine-tuning
        background_tasks.add_task(_train_auto, timestamp)

        return JSONResponse({
            "message": "✅ Image labeled and fine-tuning started.",
            "mode": "manual" if label_name else "auto",
            "image": image_path,
            "label_file": label_path,
            "label_name": label_name or "auto-detected",
            "classes": class_names,
            "train_session": f"train_{timestamp}"
        })

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

@app.get("/train/status")
def train_status():
    logs_dir = os.path.join(BASE_DIR, "runs", "auto_train")
    sessions = [d for d in os.listdir(logs_dir) if d.startswith("train_")]
    if not sessions:
        return {"status": "no training sessions yet"}

    latest = sorted(sessions)[-1]
    log_file = os.path.join(logs_dir, latest, "results.txt")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()[-10:]
        return {"session": latest, "recent_logs": lines}
    return {"session": latest, "status": "no logs yet"}


# ---------------------------------
# 🌐 REALTIME DETECTION (WebSocket)
# ---------------------------------
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    print("📡 WebSocket client connected for real-time YOLO detection")

    try:
        while True:
            data = await websocket.receive_text()
            image_bytes = base64.b64decode(data.split(",")[1])
            np_img = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            h, w, _ = frame.shape

            results = yolo(frame)
            detections = []

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (x2 - x1) > 0.9 * w and (y2 - y1) > 0.9 * h:
                        continue
                    cls = int(box.cls[0])
                    label = yolo.names[cls]
                    detections.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2]
                    })

            await websocket.send_json({"detections": detections})

    except Exception as e:
        print("❌ WebSocket error:", e)
    finally:
        await websocket.close()
        print("🛑 WebSocket disconnected")
