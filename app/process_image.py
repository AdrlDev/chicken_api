# process_image.py

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel
from app.config import (
    DATASET_DIR,
    LABELS_DIR,
    CLASSES_PATH
)
from app.train_model import _train
from app.label_studio import get_client
import cv2
from app.utils import ModelManager, PUBLIC_IMAGE_DIR

# Minimum contour area to consider a valid object
MIN_OBJECT_AREA = 500
MAX_IMAGE_DIM = 1024

processing_tasks: Dict[str, Dict] = {}

class AutoLabelResponse(BaseModel):
    message: str
    mode: str
    image: str
    label_file: str
    label_name: str
    classes: List[str]


async def process_image(task_id: str, image_path: str, label_name: str):
    try:
        ls_client = get_client()

        # -------------------- CLASSES --------------------
        classes = []
        if CLASSES_PATH.exists():
            with open(CLASSES_PATH, "r") as f:
                classes = [line.strip() for line in f if line.strip()]

        # Normalize for comparison (case-insensitive)
        classes_lower = [c.lower() for c in classes]
        label_lower = label_name.lower() if label_name else None

        # Add label if new
        if label_lower and label_lower not in classes_lower:
            classes.append(label_name)
            classes_lower.append(label_lower)

        # Remove labels not in frontend (optional, if you want syncing)
        # classes = [c for c in classes if c.lower() in classes_lower]

        # Save updated classes.txt
        with open(CLASSES_PATH, "w") as f:
            f.write("\n".join(classes) + "\n")

        # Get the index after update
        label_index = classes_lower.index(label_lower) if label_lower else 0

        # -------------------- LOAD IMAGE --------------------
        orig_img = cv2.imread(str(image_path))
        if orig_img is None:
            raise ValueError("Failed to read uploaded image")
        orig_h, orig_w = orig_img.shape[:2]

        # Resize if too large
        scale = min(1.0, MAX_IMAGE_DIM / max(orig_w, orig_h))
        img = cv2.resize(orig_img, (int(orig_w*scale), int(orig_h*scale))) if scale < 1.0 else orig_img.copy()
        h, w = img.shape[:2]

        # -------------------- DETECT CHICKENS USING BASE YOLOv8n --------------------
        base_model = ModelManager.get_base_yolov8n()
        results = base_model.predict(img, conf=0.3, save=False)  # adjust confidence if needed

        detections = []
        for r in results:
            for box in r.boxes: # type: ignore
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                # scale box back to original image size
                if scale < 1.0:
                    x1 /= scale
                    y1 /= scale
                    x2 /= scale
                    y2 /= scale
                w_box = x2 - x1
                h_box = y2 - y1
                x_center = (x1 + w_box/2) / orig_w
                y_center = (y1 + h_box/2) / orig_h
                w_norm = w_box / orig_w
                h_norm = h_box / orig_h
                detections.append({
                    "label": label_name,  # Use frontend label
                    "bbox": [x_center, y_center, w_norm, h_norm],
                    "abs_bbox": [x1, y1, w_box, h_box]
                })

        if not detections:
            processing_tasks[task_id] = {
                "status": "error",
                "error": "No chickens detected — please label manually in Label Studio"
            }
            return

        # -------------------- MOVE IMAGE --------------------
        img_filename = Path(image_path).name
        dataset_img = Path(DATASET_DIR) / "images" / img_filename
        Path(dataset_img).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(image_path, dataset_img)

        public_img = PUBLIC_IMAGE_DIR / img_filename
        shutil.copy(dataset_img, public_img)
        image_url = f"https://aedev.cloud/dataset/images/{img_filename}"

        # -------------------- CREATE LABEL STUDIO TASK --------------------
        PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))
        ls_annotations = [
            {
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": det["bbox"][0]*100 - det["bbox"][2]*50,
                    "y": det["bbox"][1]*100 - det["bbox"][3]*50,
                    "width": det["bbox"][2]*100,
                    "height": det["bbox"][3]*100,
                    "rotation": 0,
                    "rectanglelabels": [det["label"]],
                }
            }
            for det in detections
        ]

        task = await ls_client.tasks.create(data={"image": image_url}, project=PROJECT_ID)
        await ls_client.predictions.create(
            task=task.id,
            model_version="v1-auto",
            result=ls_annotations
        )

        # -------------------- SAVE YOLO LABEL FILE --------------------
        yolo_label_path = LABELS_DIR / f"{Path(image_path).stem}.txt"
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(yolo_label_path, "w") as f:
            for det in detections:
                x, y, w_box, h_box = det["bbox"]
                f.write(f"{label_index} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")

        # -------------------- UPDATE NOTES --------------------
        notes_path = Path(DATASET_DIR) / "notes.json"
        notes = {}
        if notes_path.exists():
            with open(notes_path, "r") as f:
                notes = json.load(f)
        notes[img_filename] = {
            "label": label_name,
            "upload_date": datetime.now().isoformat(),
            "detections": len(detections)
        }
        with open(notes_path, "w") as f:
            json.dump(notes, f, indent=4)
    

        # -------------------- TRAIN --------------------
        _train()

        processing_tasks[task_id] = {
            "status": "completed",
            "result": AutoLabelResponse(
                message=f"✅ {len(detections)} chickens detected and labeled with '{label_name}'.",
                mode="auto",
                image=str(dataset_img),
                label_file=str(yolo_label_path),
                label_name=label_name,
                classes=classes
            )
        }

    except Exception as e:
        processing_tasks[task_id] = {"status": "error", "error": str(e)}
