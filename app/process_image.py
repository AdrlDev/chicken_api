# app/process_image.py

import os
import json
import cv2
from pathlib import Path
from datetime import datetime
from typing import List
from app.config import DATASET_DIR, LABELS_DIR, CLASSES_PATH, CONFIDENCE_THRESHOLD, AUTO_TRAIN_EPOCHS, AUTO_TRAIN_IMAGE_SIZE
from app.train_model import _train_auto
from app.utils import yolo  # Make sure yolo is callable
from app.label_studio import get_client
from pydantic import BaseModel
from typing import List, Dict

# Dictionary to store processing status
processing_tasks: Dict[str, Dict] = {}

# Response models for auto-label
class AutoLabelResponse(BaseModel):
    message: str
    mode: str
    image: str
    label_file: str
    label_name: str
    classes: List[str]

async def process_image(task_id: str, image_path: str, label_name: str):
    """
    Process an uploaded image:
    - Detect chickens using YOLO
    - Save YOLO label files locally
    - Send pre-annotations to Label Studio
    - Update dataset metadata
    - Optionally trigger auto-training
    """
    try:
        # Initialize Label Studio client
        ls_client = get_client()

        # Load existing classes
        class_names = []
        if CLASSES_PATH.exists():
            with open(CLASSES_PATH, "r") as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]

        # Add label if missing
        if label_name not in class_names:
            class_names.append(label_name)
            with open(CLASSES_PATH, "w") as f:
                f.write("\n".join(class_names))

        label_index = class_names.index(label_name)
        detections: List[dict] = []

        # Run YOLO detection
        results = yolo.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, save=False)  # type: ignore
        if not results or len(results[0].boxes) == 0:  # type: ignore
            processing_tasks[task_id]["status"] = "error"
            processing_tasks[task_id]["error"] = "No chickens detected in the image"
            return

        for box in results[0].boxes:  # type: ignore
            x_center, y_center, width, height = box.xywhn[0].tolist()
            confidence = float(box.conf[0])
            detections.append({
                "label": label_name,
                "confidence": confidence,
                "bbox": [x_center, y_center, width, height]
            })

        # Save YOLO label file
        image_filename = Path(image_path).name
        label_filename = Path(image_path).stem + ".txt"
        label_file_path = LABELS_DIR / label_filename
        with open(label_file_path, "w") as f:
            for det in detections:
                f.write(f"{label_index} {det['bbox'][0]:.6f} {det['bbox'][1]:.6f} {det['bbox'][2]:.6f} {det['bbox'][3]:.6f}\n")

        # Move image into dataset/images
        dataset_img_path = Path(DATASET_DIR) / "images" / image_filename
        Path(dataset_img_path).parent.mkdir(parents=True, exist_ok=True)
        os.rename(image_path, dataset_img_path)

        # -------------------------------
        # Prepare Label Studio pre-annotation
        # -------------------------------
        image_url = f"https://aedev.cloud/dataset/images/{image_filename}"

        ls_tasks = [
            {
                "data": {"image": image_url},  # or URL if LS requires
                "predictions": [
                    {
                        "model_version": "v1",  # optional
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "image",
                                "type": "rectanglelabels",
                                "value": {
                                    "x": (det["bbox"][0] - det["bbox"][2] / 2) * 100,
                                    "y": (det["bbox"][1] - det["bbox"][3] / 2) * 100,
                                    "width": det["bbox"][2] * 100,
                                    "height": det["bbox"][3] * 100,
                                    "rotation": 0,
                                    "rectanglelabels": [det["label"]],
                                },
                            }
                            for det in detections
                        ],
                    }
                ],
            }
        ]

        PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))
        # Prepare tasks with the required project ID
        # Create tasks asynchronously
        for t in ls_tasks:
            try:
                response = await ls_client.tasks.create(
                    data=t["data"],
                    project=PROJECT_ID
                )
                print("✅ Task created:", response)
            except Exception as e:
                print("⚠️ Failed to upload to Label Studio:", str(e))
        # -------------------------------
        # Update dataset metadata (notes.json)
        # -------------------------------
        notes_path = Path(DATASET_DIR) / "notes.json"
        notes = {}
        if notes_path.exists():
            with open(notes_path, "r") as f:
                notes = json.load(f)

        notes[image_filename] = {
            "label": label_name,
            "upload_date": datetime.now().isoformat(),
            "detections": len(detections)
        }

        with open(notes_path, "w") as f:
            json.dump(notes, f, indent=4)

        # -------------------------------
        # Update processing task status
        # -------------------------------
        processing_tasks[task_id].update({
            "status": "completed",
            "result": AutoLabelResponse(
                message="✅ Image labeled successfully",
                mode="auto",
                image=str(dataset_img_path),
                label_file=str(label_file_path),
                label_name=label_name,
                classes=class_names
            )
        })

        # -------------------------------
        # Trigger auto training
        # -------------------------------
        _train_auto(epochs=AUTO_TRAIN_EPOCHS, imgsz=AUTO_TRAIN_IMAGE_SIZE)

    except Exception as e:
        processing_tasks[task_id].update({
            "status": "error",
            "error": str(e)
        })
