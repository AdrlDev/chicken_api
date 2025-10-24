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
TRAINED_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")

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