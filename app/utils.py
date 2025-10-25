#utils.py

import os
from ultralytics import YOLO  # type: ignore
from datetime import datetime

# ---------------------------------
# 🔧 PATH CONFIGURATION
# ---------------------------------
BASE_DIR = os.path.dirname(__file__)
YOLO_WEIGHTS = os.path.join(BASE_DIR, "assets", "yolov8n.pt")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
# Dynamically find the most recent YOLO trained weights
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")

def get_latest_trained_weights() -> str:
    if not os.path.exists(RUNS_DIR):
        return YOLO_WEIGHTS
    subdirs = [os.path.join(RUNS_DIR, d) for d in os.listdir(RUNS_DIR)
               if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.startswith("train")]
    if not subdirs:
        return YOLO_WEIGHTS
    latest_run = max(subdirs, key=os.path.getmtime)
    best_path = os.path.join(latest_run, "weights", "best.pt")
    return best_path if os.path.exists(best_path) else YOLO_WEIGHTS

TRAINED_WEIGHTS = get_latest_trained_weights()

# YOLO dataset structure
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

# ---------------------------------
# 🧠 LOAD INITIAL YOLO MODEL
# ---------------------------------
yolo = YOLO(TRAINED_WEIGHTS if os.path.exists(TRAINED_WEIGHTS) else YOLO_WEIGHTS)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_filename = f"auto_{timestamp}.jpg"
image_path = os.path.join(IMAGES_DIR, image_filename)
classes_path = os.path.join(DATASET_DIR, "classes.txt")