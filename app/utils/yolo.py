# app/utils/yolo.py (Hypothetical modification)

from ultralytics import YOLO

# 💡 CHANGE: Define yolo as a global variable, but DO NOT load the model here.
# The model object will be assigned in the FastAPI lifespan function.
global yolo_model
yolo_model = None

def load_yolo_model():
    """Function to load the model."""
    global yolo_model
    if yolo_model is None:
        print("🔄 Loading YOLO model in a safe function...")
        # Replace this path with your actual model loading logic
        yolo_model = YOLO("/root/chicken_api/runs/detect/train/weights/best.pt")
        print("✅ YOLO model loaded.")
    return yolo_model