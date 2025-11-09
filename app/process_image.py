# app/process_image.py

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
    CONFIDENCE_THRESHOLD,
    AUTO_TRAIN_EPOCHS,
    AUTO_TRAIN_IMAGE_SIZE,
)
from app.train_model import _train_auto
from app.utils import yolo  # YOLO model callable
from app.label_studio import get_client

# Where Nginx serves public images
PUBLIC_IMAGE_DIR = Path("/var/www/chicken_api/dataset/images")

# Dictionary to track async processing jobs
processing_tasks: Dict[str, Dict] = {}


# Response model
class AutoLabelResponse(BaseModel):
    message: str
    mode: str
    image: str
    label_file: str
    label_name: str
    classes: List[str]


async def process_image(task_id: str, image_path: str, label_name: str):
    """
    1️⃣ Detect chickens in uploaded image using YOLO
    2️⃣ Create Label Studio task with bounding boxes (auto annotations)
    3️⃣ Save YOLO-format label file in dataset
    4️⃣ Automatically trigger training with the new data
    """
    try:
        ls_client = get_client()

        # ------------------------------
        # 1️⃣ Load classes
        # ------------------------------
        class_names = []
        if CLASSES_PATH.exists():
            with open(CLASSES_PATH, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]

        # Add class if not exist
        if label_name not in class_names:
            class_names.append(label_name)
            with open(CLASSES_PATH, "w") as f:
                f.write("\n".join(class_names))

        label_index = class_names.index(label_name)

        # ------------------------------
        # 2️⃣ Run YOLO detection
        # ------------------------------
        results = yolo.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, save=False)  # type: ignore
        if not results or len(results[0].boxes) == 0:  # type: ignore
            processing_tasks[task_id]["status"] = "error"
            processing_tasks[task_id]["error"] = "No chickens detected in the image"
            return

        detections = []
        for box in results[0].boxes:  # type: ignore
            x_center, y_center, width, height = box.xywhn[0].tolist()
            confidence = float(box.conf[0])
            detections.append({
                "label": label_name,
                "confidence": confidence,
                "bbox": [x_center, y_center, width, height]
            })

        # ------------------------------
        # 3️⃣ Save YOLO label file
        # ------------------------------
        image_filename = Path(image_path).name
        label_filename = Path(image_path).stem + ".txt"
        label_file_path = LABELS_DIR / label_filename
        LABELS_DIR.mkdir(parents=True, exist_ok=True)

        with open(label_file_path, "w") as f:
            for det in detections:
                f.write(
                    f"{label_index} "
                    f"{det['bbox'][0]:.6f} {det['bbox'][1]:.6f} "
                    f"{det['bbox'][2]:.6f} {det['bbox'][3]:.6f}\n"
                )

        # ------------------------------
        # 4️⃣ Move image into dataset
        # ------------------------------
        dataset_img_path = Path(DATASET_DIR) / "images" / image_filename
        Path(dataset_img_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(image_path, dataset_img_path)

        # Copy to public folder for Label Studio
        public_img_path = PUBLIC_IMAGE_DIR / image_filename
        shutil.copy(dataset_img_path, public_img_path)
        image_url = f"https://aedev.cloud/dataset/images/{image_filename}"

        # ------------------------------
        # 5️⃣ Create Label Studio task
        # ------------------------------
        PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

        ls_task_data = {
            "data": {"image": image_url},
            "predictions": [
                {
                    "model_version": "auto-chicken-detector-v1",
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

        # Create task
        task = await ls_client.tasks.create(data=ls_task_data["data"], project=PROJECT_ID)

        # Add pre-annotations
        await ls_client.predictions.create(
            task=task.id,
            model_version=ls_task_data["predictions"][0]["model_version"],
            result=ls_task_data["predictions"][0]["result"]
        )

        # ------------------------------
        # 6️⃣ Update notes.json
        # ------------------------------
        notes_path = Path(DATASET_DIR) / "notes.json"
        notes = {}
        if notes_path.exists():
            with open(notes_path, "r") as f:
                notes = json.load(f)

        notes[image_filename] = {
            "label": label_name,
            "upload_date": datetime.now().isoformat(),
            "detections": len(detections),
        }

        with open(notes_path, "w") as f:
            json.dump(notes, f, indent=4)

        # ------------------------------
        # 7️⃣ Trigger auto training
        # ------------------------------
        _train_auto(epochs=AUTO_TRAIN_EPOCHS, imgsz=AUTO_TRAIN_IMAGE_SIZE)

        # ------------------------------
        # 8️⃣ Update processing status
        # ------------------------------
        processing_tasks[task_id].update({
            "status": "completed",
            "result": AutoLabelResponse(
                message=f"✅ {len(detections)} chickens detected and task created in Label Studio",
                mode="auto",
                image=str(dataset_img_path),
                label_file=str(label_file_path),
                label_name=label_name,
                classes=class_names
            )
        })

    except Exception as e:
        processing_tasks[task_id].update({
            "status": "error",
            "error": str(e)
        })
