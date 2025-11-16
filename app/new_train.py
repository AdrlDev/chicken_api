# new_train.py
import os
import threading
import asyncio
from datetime import datetime
from ultralytics import YOLO
import app.utils as utils

# Thread lock for safe YOLO reloading
reload_lock = threading.Lock()

async def train_yolo_autosplit_ws(ws=None, dataset_dir=None, epochs=50, imgsz=640):
    """
    Train YOLO and send live logs to ws manager (can be None)
    """
    async def log(msg: str):
        print(msg)  # also print in console
        if ws:
            await ws.send(msg)

    await log(f"🚀 Training started at {datetime.now()}")

    weights_path = utils.get_latest_trained_weights()
    model = YOLO(weights_path)

    if not dataset_dir:
        raise ValueError("dataset_dir cannot be None")

    data_yaml = os.path.join(dataset_dir, "data.yaml")

    class WSLogger:
        def __call__(self, info):
            # info is usually a dict or string from ultralytics
            msg = str(info)
            if ws:
                asyncio.create_task(ws.send(msg))
            else:
                print(msg)

    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            project=os.path.join(utils.BASE_DIR, "runs", "detect"),
            name="train",
            exist_ok=True,
            batch=1,
            verbose=True,
            callbacks=[WSLogger()]
        )

        await log(f"✅ Training finished at {datetime.now()}")

        # Reload model safely
        latest_weights = utils.get_latest_trained_weights()
        global yolo
        yolo = YOLO(latest_weights)
        await log(f"✅ Model reloaded with new weights: {latest_weights}")

    except Exception as e:
        await log(f"❌ Training failed: {str(e)}")

