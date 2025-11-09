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
    CLASSES_PATH,
    AUTO_TRAIN_EPOCHS,
    AUTO_TRAIN_IMAGE_SIZE,
)
from app.train_model import _train_auto
from app.label_studio import get_client
import cv2
import numpy as np
from PIL import Image

# Minimum contour area to consider a valid object (adjust based on your images)
MIN_OBJECT_AREA = 500  
MAX_IMAGE_DIM = 1024  # Maximum width or height for resizing

PUBLIC_IMAGE_DIR = Path("/var/www/chicken_api/dataset/images")
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
        if label_name not in classes:
            classes.append(label_name)
            with open(CLASSES_PATH, "w") as f:
                f.write("\n".join(classes))
        label_index = classes.index(label_name)

        # -------------------- LOAD IMAGE --------------------
        orig_img = cv2.imread(str(image_path))
        if orig_img is None:
            raise ValueError("Failed to read uploaded image")

        orig_h, orig_w = orig_img.shape[:2]

        # Resize if too large
        scale = min(1.0, MAX_IMAGE_DIM / max(orig_w, orig_h))
        if scale < 1.0:
            img = cv2.resize(orig_img, (int(orig_w*scale), int(orig_h*scale)))
        else:
            img = orig_img.copy()

        h, w = img.shape[:2]

        # -------------------- CONTOUR DETECTION --------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            processing_tasks[task_id] = {
                "status": "error",
                "error": "No objects detected — please label manually in Label Studio"
            }
            return

        # -------------------- CREATE DETECTIONS --------------------
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_OBJECT_AREA:
                continue
            x, y, w_box, h_box = cv2.boundingRect(cnt)

            # Scale bbox back to original image size
            if scale < 1.0:
                x = x / scale
                y = y / scale
                w_box = w_box / scale
                h_box = h_box / scale

            x_center = (x + w_box / 2) / orig_w
            y_center = (y + h_box / 2) / orig_h
            w_norm = w_box / orig_w
            h_norm = h_box / orig_h

            detections.append({
                "label": label_name,
                "bbox": [x_center, y_center, w_norm, h_norm],
                "abs_bbox": [x, y, w_box, h_box]
            })

        if not detections:
            processing_tasks[task_id] = {
                "status": "error",
                "error": "No objects above minimum size detected — please label manually"
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
        _train_auto(epochs=AUTO_TRAIN_EPOCHS, imgsz=AUTO_TRAIN_IMAGE_SIZE)

        processing_tasks[task_id] = {
            "status": "completed",
            "result": AutoLabelResponse(
                message=f"✅ {len(detections)} objects detected and labeled automatically, retraining started",
                mode="auto",
                image=str(dataset_img),
                label_file=str(yolo_label_path),
                label_name=label_name,
                classes=classes
            )
        }

    except Exception as e:
        processing_tasks[task_id] = {"status": "error", "error": str(e)}