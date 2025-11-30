# process_image.py (FIXED)

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel
from app.utils.config import (
    DATASET_DIR,
    LABELS_DIR,
    CLASSES_PATH,
    PUBLIC_IMAGE_DIR,
    YOLO_WEIGHTS
)
from app.label_studio import get_client
import cv2
from ultralytics import YOLO

# Minimum contour area to consider a valid object
MIN_OBJECT_AREA = 500
MAX_IMAGE_DIM = 1024

processing_tasks: Dict[str, Dict] = {}

# --- NEW: Defined classes from your Label Studio config ---
DEFAULT_CLASSES = [
    "healthy",
    "avian Influenza",
    "blue comb",
    "coccidiosis",
    "coccidiosis poops",
    "fowl cholera",
    "fowl-pox",
    "mycotic infections",
    "salmo",
]
# --------------------------------------------------------

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
        CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        needs_write = False
        
        # 1. Load Existing Classes
        if CLASSES_PATH.exists() and os.stat(CLASSES_PATH).st_size > 0:
            with open(CLASSES_PATH, "r") as f:
                classes = [line.strip() for line in f if line.strip()]

        # 2. Enforce DEFAULT_CLASSES (Guaranteed Save)
        if classes != DEFAULT_CLASSES:
            classes = DEFAULT_CLASSES
            needs_write = True

        # ⭐️ FIX: Define classes_lower here, using the guaranteed 'classes' list
        classes_lower = [c.lower() for c in classes] # <-- NOW ALWAYS DEFINED

        # 3. Check and Add New Label
        label_lower = label_name.lower() if label_name else None
        
        if label_lower:
            # classes_lower is already available here, no need to redefine unless we change 'classes'
            if label_lower not in classes_lower:
                # The label name from parameters is new, so add it.
                classes.append(label_name)
                needs_write = True
                
                # Update classes_lower immediately after adding a new class
                classes_lower = [c.lower() for c in classes] # Update for correct index look-up


        # 4. Save to File System
        if needs_write: 
            with open(CLASSES_PATH, "w") as f:
                f.write("\n".join(classes) + "\n")

        
        # 5. Final Index Look-up
        if not label_lower or label_lower not in classes_lower:
            # This check now safely uses the always-defined classes_lower
            raise ValueError(f"Label name '{label_name}' is invalid or missing.")

        label_index = classes_lower.index(label_lower)

        # -------------------- CLASSES --------------------
        # ... (Classes logic remains the same, assuming it's correct) ...
        # -------------------- LOAD IMAGE & DETECT CHICKENS --------------------
        orig_img = cv2.imread(str(image_path))
        if orig_img is None:
            raise ValueError("Failed to read uploaded image")
        orig_h, orig_w = orig_img.shape[:2]

        # Resize if too large
        scale = min(1.0, MAX_IMAGE_DIM / max(orig_w, orig_h))
        img = cv2.resize(orig_img, (int(orig_w*scale), int(orig_h*scale))) if scale < 1.0 else orig_img.copy()

        # DETECT CHICKENS USING BASE YOLOv8n
        base_model = YOLO(YOLO_WEIGHTS)
        results = base_model.predict(img, conf=0.3, save=False)

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
                    "label": label_name,
                    "bbox": [x_center, y_center, w_norm, h_norm],
                    "abs_bbox": [x1, y1, w_box, h_box]
                })

        # -------------------- MOVE IMAGE TO FINAL LOCATION & PUBLIC DIR --------------------
        original_stem = Path(image_path).stem
        
        # Ensure image is moved to the final dataset directory regardless of detection success
        dataset_img = Path(DATASET_DIR) / "images" / Path(image_path).name
        dataset_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(image_path, dataset_img) # Move the image to final dataset/images

        # Copy to public directory for Label Studio access
        public_img = PUBLIC_IMAGE_DIR / dataset_img.name
        public_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(dataset_img, public_img)
        image_url = f"https://aedev.cloud/dataset/images/{dataset_img.name}"
        
        # -------------------- HANDLING ZERO DETECTIONS --------------------
        if not detections:
            # ⭐ FIX: Do NOT return here. Continue to create the LS task for manual annotation.
            processing_tasks[task_id] = {
                "status": "completed", # Mark as completed, but with a warning/manual mode
                "result": AutoLabelResponse(
                    message=f"⚠️ No chickens detected. Task created for manual labeling.",
                    mode="manual",
                    image=str(dataset_img),
                    label_file="N/A", # No label file created
                    label_name=label_name,
                    classes=classes
                )
            }
            # Skip label file creation and prediction, but create the base task.
            
        # -------------------- CREATE YOLO LABEL FILE (Only if detections exist) --------------------
        yolo_label_path = LABELS_DIR / f"{original_stem}.txt"
        
        if detections:
            temp_yolo_label_path = Path(dataset_img.parent) / f"{original_stem}.txt"
            
            # 1. Create temporary YOLO label file
            with open(temp_yolo_label_path, "w") as f:
                for det in detections:
                    x, y, w_box, h_box = det["bbox"]
                    f.write(f"{label_index} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")

            # 2. Move label file to final location
            LABELS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(temp_yolo_label_path, yolo_label_path) # Move the label file to final dataset/labels

        # -------------------- CREATE LABEL STUDIO TASK --------------------
        PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "7"))
        
        # FIX: Define empty annotations if no detections, so prediction creation is skipped.
        ls_annotations = []
        if detections:
            ls_annotations = [
                {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        # ... (Existing geometry calculations for Label Studio) ...
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

        # ⭐ CRITICAL STEP: Create the task regardless of detections
        task = await ls_client.tasks.create(data={"image": image_url}, project=PROJECT_ID)

        # Only create a prediction if detections were made
        if ls_annotations:
            await ls_client.predictions.create(
                task=task.id,
                model_version="v1-auto",
                result=ls_annotations
            )

            # Update the task status result if detections were made
            processing_tasks[task_id]["result"] = AutoLabelResponse(
                message=f"✅ {len(detections)} chickens detected and labeled with '{label_name}'.",
                mode="auto",
                image=str(dataset_img),
                label_file=str(yolo_label_path),
                label_name=label_name,
                classes=classes
            )

        # -------------------- UPDATE NOTES --------------------
        notes_path = Path(DATASET_DIR) / "notes.json"
        notes = {}
        if notes_path.exists():
            with open(notes_path, "r") as f:
                notes = json.load(f)
        notes[dataset_img.name] = {
            "label": label_name,
            "upload_date": datetime.now().isoformat(),
            "detections": len(detections), # Will be 0 if skipped prediction
            "ls_task_id": task.id # Added LS Task ID for tracking
        }
        with open(notes_path, "w") as f:
            json.dump(notes, f, indent=4)
        
        # If no detections were made, the `processing_tasks` result message 
        # set earlier will remain, indicating it needs manual labeling.

    except Exception as e:
        processing_tasks[task_id] = {"status": "error", "error": str(e)}