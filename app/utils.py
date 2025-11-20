# utils.py
import os
from pathlib import Path
from ultralytics import YOLO  # type: ignore
from datetime import datetime
import torch
from app.config import (
    BASE_DIR,
    YOLO_WEIGHTS
)

# ---------------------------------
# 🔄 FUNCTION TO GET LATEST TRAINED WEIGHTS
# ---------------------------------
def get_latest_trained_weights() -> str:
    """Returns the most recent trained best.pt, else fall back to YOLO_WEIGHTS"""
    detect_dir = BASE_DIR / "runs" / "detect"
    if not detect_dir.exists():
        return str(YOLO_WEIGHTS)

    # Find all train folders
    train_folders = [d for d in detect_dir.iterdir() if d.is_dir() and d.name.startswith("train")]
    if not train_folders:
        return str(YOLO_WEIGHTS)

    # Pick the latest folder (by modification time)
    latest_train = max(train_folders, key=lambda d: d.stat().st_mtime)
    best_pt = latest_train / "weights" / "best.pt"

    if best_pt.exists():
        print(f"📌 Found latest trained model: {best_pt}")
        return str(best_pt)

    return str(YOLO_WEIGHTS)

# ---------------------------------
# 🧠 YOLO MODEL MANAGER
# ---------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("⚠️ Warning: Running on CPU. For better performance, use a GPU.")

class ModelManager:
    _instance = None
    _last_weights_path = None

    @classmethod
    def get_model(cls, force_reload=False):
        latest_weights = get_latest_trained_weights()
        if cls._instance is None or force_reload or (latest_weights != cls._last_weights_path):
            print(f"🔄 Loading YOLO model from: {latest_weights}")
            cls._instance = YOLO(latest_weights)
            cls._instance.to(DEVICE)
            cls._last_weights_path = latest_weights
            print(f"✅ Model loaded successfully on {DEVICE}")
        return cls._instance

    @classmethod
    def get_base_yolov8n(cls, force_reload=False):
        """
        Always load the base YOLOv8n model (yolov8n.pt), ignoring any trained weights.
        """
        print(f"🔄 Loading base YOLOv8n model from: {YOLO_WEIGHTS}")
        base_model = YOLO(YOLO_WEIGHTS)
        base_model.to(DEVICE)
        print(f"✅ Base YOLOv8n model loaded successfully on {DEVICE}")
        return base_model

# Create singleton instance for general use
yolo = ModelManager.get_model(force_reload=True)
yoloV8n = ModelManager.get_base_yolov8n()
