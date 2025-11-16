# utils.py
import os
from pathlib import Path
from ultralytics import YOLO  # type: ignore
from datetime import datetime
import torch

# ---------------------------------
# 🔧 PATH CONFIGURATION
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Go up one level to project root
YOLO_WEIGHTS = os.path.join(BASE_DIR, "assets", "yolov8n.pt")  # Initial YOLOv8n weights
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")
PUBLIC_IMAGE_DIR = Path("/var/www/chicken_api/dataset/images")

# ---------------------------------
# 🔄 FUNCTION TO GET LATEST TRAINED WEIGHTS
# ---------------------------------
def get_latest_trained_weights() -> str:
    """Returns the most recent trained best.pt, else fall back to assets."""
    save_dir = os.path.join(BASE_DIR, "runs", "detect", "train", "weights")
    trained_best = os.path.join(save_dir, "best.pt")

    if os.path.exists(trained_best):
        print(f"📌 Found trained model: {trained_best}")
        return trained_best
    
    print(f"📌 No trained model found, using asset base: {YOLO_WEIGHTS}")
    return YOLO_WEIGHTS

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
        """
        Get or create YOLO model instance with latest trained weights.
        """
        latest_weights = get_latest_trained_weights()
        if cls._instance is None or force_reload or (latest_weights != cls._last_weights_path and os.path.exists(latest_weights)):
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
yolo = ModelManager.get_model()
yoloV8n = ModelManager.get_base_yolov8n()
