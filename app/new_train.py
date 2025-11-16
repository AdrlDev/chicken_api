# app/train_model.py
import os
import threading
import asyncio
from datetime import datetime
from ultralytics import YOLO
from app.utils import DATASET_DIR, BASE_DIR
import app.utils as utils

# Thread lock for safe YOLO reloading
reload_lock = threading.Lock()

async def train_yolo_autosplit_ws(ws=None, dataset_dir=DATASET_DIR, epochs=50, imgsz=640):
    """
    Train YOLO and send real-time logs via WebSocket.
    If ws=None, fallback to normal stdout logging.
    """
    log_msg = lambda msg: asyncio.create_task(ws.send_text(msg)) if ws else print(msg)

    log_msg(f"🚀 Training started at {datetime.now()}")
    
    # Use latest trained weights or base model
    weights_path = utils.get_latest_trained_weights()
    model = YOLO(weights_path)

    # Custom callback for logging
    class WSLogger:
        def __call__(self, info):
            asyncio.create_task(ws.send_text(str(info))) if ws else print(info)

    try:
        model.train(
            data=os.path.join(dataset_dir, "data.yaml"),
            epochs=epochs,
            imgsz=imgsz,
            project=os.path.join(BASE_DIR, "runs", "detect"),
            name="train",
            exist_ok=True,
            batch=1,  # safe for CPU
            verbose=True
        )

        log_msg(f"✅ Training finished at {datetime.now()}")

        # Reload model safely
        with reload_lock:
            latest_weights = utils.get_latest_trained_weights()
            global yolo
            yolo = YOLO(latest_weights)
            log_msg(f"✅ Model reloaded with new weights: {latest_weights}")

    except Exception as e:
        log_msg(f"❌ Training failed: {str(e)}")
