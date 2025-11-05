"""Configuration settings for the ChickenAI application."""

from pathlib import Path

# Model settings
CONFIDENCE_THRESHOLD = 0.4
WEBSOCKET_CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 640

# Training settings
AUTO_TRAIN_EPOCHS = 5
AUTO_TRAIN_IMAGE_SIZE = 640

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
CLASSES_PATH = DATASET_DIR / "classes.txt"
LOGS_DIR = BASE_DIR / "runs" / "auto_train"

# WebSocket settings
WS_MAX_CONNECTIONS = 5
WS_TIMEOUT = 60  # seconds