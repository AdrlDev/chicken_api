# retrain_with_roboflow.py
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
from app.utils.config import DATASET_DIR, YOLO_WEIGHTS, IMAGES_DIR, LABELS_DIR, CLASSES_PATH, ROBOFLOW
from app.utils.utils import get_latest_trained_weights
from app.utils.ws_manager import ws_manager
import asyncio
import json

# -------------------
# Device & loop
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Training on device: {device}")
MAIN_LOOP = asyncio.get_event_loop()

# -------------------
# Classes & label remap
# -------------------
CLASSES = [
    "healthy",
    "avian Influenza",
    "blue comb",
    "coccidiosis",
    "coccidiosis poops",
    "fowl cholera",
    "fowl-pox",
    "mycotic infections",
    "salmo"
]

# Remap rules for Roboflow labels
LABEL_REMAP = {
    "cocci": "coccidiosis poops",
}

# -------------------
# Paths
# -------------------
DATA_YAML = DATASET_DIR / "data.yaml"

# -------------------
# Merge Roboflow into main dataset
# -------------------
def merge_roboflow_dataset():
    for split in ["train", "valid", "test"]:
        rf_images = ROBOFLOW / split / "images"
        rf_labels = ROBOFLOW / split / "labels"

        if rf_images.exists():
            for file in rf_images.iterdir():
                dest = IMAGES_DIR / file.name
                if not dest.exists():
                    shutil.copy(file, dest)

        if rf_labels.exists():
            for file in rf_labels.iterdir():
                dest = LABELS_DIR / file.name
                if dest.exists():
                    continue

                # Read original labels
                with open(file, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    # Map Roboflow class name to your classes
                    # Assume Roboflow uses same order as names in data.yaml
                    rf_class_name = parts[1] if len(parts) > 1 else None  # adjust if necessary
                    if rf_class_name is None:
                        continue  # skip invalid label
                    mapped_name = LABEL_REMAP.get(rf_class_name, rf_class_name)
                    if mapped_name in CLASSES:
                        # Convert back to class index
                        class_idx = CLASSES.index(mapped_name)
                        parts[0] = str(class_idx)
                        new_lines.append(" ".join(parts))

                # Write filtered/remapped labels
                if new_lines:
                    with open(dest, "w") as f:
                        f.write("\n".join(new_lines) + "\n")

    print(f"📥 Roboflow dataset merged and remapped into {IMAGES_DIR} and {LABELS_DIR}")

# -------------------
# Update classes.txt
# -------------------
def update_classes_txt():
    with open(CLASSES_PATH, "w") as f:
        for cls in CLASSES:
            f.write(cls + "\n")
    print(f"✅ classes.txt updated with {len(CLASSES)} classes")

# -------------------
# Update data.yaml
# -------------------
def update_data_yaml():
    yaml_content = (
        f"train: {IMAGES_DIR}/train\n"
        f"val: {IMAGES_DIR}/val\n"
        f"test: {IMAGES_DIR}/test\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    with open(DATA_YAML, "w") as f:
        f.write(yaml_content)
    print(f"✅ data.yaml updated")

# -------------------
# Train YOLO
# -------------------
def train_yolo(epochs=50, imgsz=640):
    weights_to_use = get_latest_trained_weights()
    if weights_to_use is None or not Path(weights_to_use).exists():
        print("⚠️ Latest trained weights not found, using base YOLO weights")
        weights_to_use = YOLO_WEIGHTS

    model = YOLO(weights_to_use)
    model.to(device)

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=1,
        project="runs/detect",
        name="train",
        exist_ok=True
    )

    print("✅ Training finished")
    latest = get_latest_trained_weights()
    return latest if latest is not None else YOLO_WEIGHTS

# -------------------
# WS helper
# -------------------
def send_ws_event(event: dict):
    try:
        msg = json.dumps(event)
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(msg), MAIN_LOOP)
    except Exception as e:
        print("❌ Failed to send WS event:", e)

# -------------------
# Main retrain function
# -------------------
def retrain_with_roboflow():
    merge_roboflow_dataset()
    update_classes_txt()
    update_data_yaml()
    weights = train_yolo()
    send_ws_event({"event": "training_finished", "weights": str(weights)})
    print(f"🎯 Model updated and saved at {weights}")

# -------------------
# Run
# -------------------
if __name__ == "__main__":
    retrain_with_roboflow()
