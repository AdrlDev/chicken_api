#utils.py

import os
from ultralytics import YOLO  # type: ignore
from datetime import datetime

# ---------------------------------
# 🔧 PATH CONFIGURATION
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Go up one level to project root
YOLO_WEIGHTS = os.path.join(BASE_DIR, "assets", "yolov8n.pt")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
# Dynamically find the most recent YOLO trained weights
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")

def get_latest_trained_weights() -> str:
    """
    Get the path to the most recently trained weights file.
    Returns the path to the latest best.pt file, or YOLO_WEIGHTS if no trained weights exist.
    """
    try:
        if not os.path.exists(RUNS_DIR):
            print("ℹ️ No trained weights found, using initial weights:", YOLO_WEIGHTS)
            return YOLO_WEIGHTS

        # Get all training directories sorted by modification time (newest first)
        train_dirs = [
            d for d in os.listdir(RUNS_DIR)
            if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.startswith("train")
        ]
        
        if not train_dirs:
            print("ℹ️ No training runs found, using initial weights:", YOLO_WEIGHTS)
            return YOLO_WEIGHTS

        # Sort by modification time, newest first
        train_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(RUNS_DIR, x)), reverse=True)
        
        # Look for best.pt in each directory until we find one
        for train_dir in train_dirs:
            weights_dir = os.path.join(RUNS_DIR, train_dir, "weights")
            best_path = os.path.join(weights_dir, "best.pt")
            last_path = os.path.join(weights_dir, "last.pt")
            
            if os.path.exists(best_path):
                print("✅ Using latest trained weights:", best_path)
                print(f"   From training run: {train_dir}")
                return best_path
            elif os.path.exists(last_path):
                print("ℹ️ No best.pt found, using last.pt from:", last_path)
                print(f"   From training run: {train_dir}")
                return last_path

        print("⚠️ No valid weights found in training directories, using initial weights:", YOLO_WEIGHTS)
        return YOLO_WEIGHTS
        
    except Exception as e:
        print(f"⚠️ Error finding latest weights: {str(e)}")
        print("ℹ️ Falling back to initial weights:", YOLO_WEIGHTS)
        return YOLO_WEIGHTS

TRAINED_WEIGHTS = get_latest_trained_weights()

# YOLO dataset structure
IMAGES_DIR = os.path.join(DATASET_DIR, "images")  # Parent directory for all images
LABELS_DIR = os.path.join(DATASET_DIR, "labels")  # Parent directory for all labels

# Training directories
TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, "train")
TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, "train")
VAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "val")
VAL_LABELS_DIR = os.path.join(LABELS_DIR, "val")

# Create all required directories
for d in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------
# 🧠 LOAD INITIAL YOLO MODEL
# ---------------------------------
import torch

# Check CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("⚠️ Warning: Running on CPU. For better performance, consider using a GPU.")

# Initialize model (singleton pattern)
class ModelManager:
    _instance = None
    _last_weights_path = None
    
    @classmethod
    def get_model(cls, force_reload=False):
        """
        Get or create the YOLO model instance.
        
        Args:
            force_reload: If True, reload the model even if instance exists
        """
        latest_weights = get_latest_trained_weights()
        
        # Reload if: no instance, forced reload, or new weights available
        if (cls._instance is None or 
            force_reload or 
            (latest_weights != cls._last_weights_path and os.path.exists(latest_weights))):
            
            try:
                print(f"🔄 Loading YOLO model from: {latest_weights}")
                cls._instance = YOLO(latest_weights)
                cls._last_weights_path = latest_weights
                
                # Set device and other parameters
                cls._instance.to(DEVICE)
                print(f"✅ Model loaded successfully on {DEVICE}")
                
                # Disable warnings about pin_memory when no GPU is available
                if DEVICE == "cpu":
                    import warnings
                    warnings.filterwarnings("ignore", message=".*pin_memory.*")
                    
            except Exception as e:
                print(f"⚠️ Error loading model from {latest_weights}: {str(e)}")
                if latest_weights != YOLO_WEIGHTS:
                    print(f"ℹ️ Falling back to initial weights: {YOLO_WEIGHTS}")
                    cls._instance = YOLO(YOLO_WEIGHTS)
                    cls._last_weights_path = YOLO_WEIGHTS
        
        return cls._instance

# Create model instance
yolo = ModelManager.get_model()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_filename = f"auto_{timestamp}.jpg"
image_path = os.path.join(IMAGES_DIR, image_filename)
classes_path = os.path.join(DATASET_DIR, "classes.txt")