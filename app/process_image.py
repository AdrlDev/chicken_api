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
from app.utils import yolo
from app.label_studio import get_client

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

        # -------------------- PRE-DETECTION --------------------
        # use YOLO only for pre-detecting chickens (initial bounding boxes)
        results = yolo.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, save=False) # type: ignore
        boxes = results[0].boxes if results and len(results[0].boxes) > 0 else [] # type: ignore

        if not boxes:
            processing_tasks[task_id] = {
                "status": "error",
                "error": "No chickens detected — please label manually in Label Studio"
            }
            return

        detections = []
        for b in boxes:
            x, y, w, h = b.xywhn[0].tolist()
            detections.append({
                "label": label_name,
                "bbox": [x, y, w, h]
            })

        # -------------------- MOVE IMAGE --------------------
        img_filename = Path(image_path).name
        dataset_img = Path(DATASET_DIR) / "images" / img_filename
        Path(dataset_img).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(image_path, dataset_img)

        # Public copy for Label Studio
        public_img = PUBLIC_IMAGE_DIR / img_filename
        shutil.copy(dataset_img, public_img)
        image_url = f"https://aedev.cloud/dataset/images/{img_filename}"

        # -------------------- CREATE TASK --------------------
        PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

        results_list = [
            {
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": (d["bbox"][0] - d["bbox"][2] / 2) * 100,
                    "y": (d["bbox"][1] - d["bbox"][3] / 2) * 100,
                    "width": d["bbox"][2] * 100,
                    "height": d["bbox"][3] * 100,
                    "rotation": 0,
                    "rectanglelabels": [d["label"]],
                },
            }
            for d in detections
        ]

        task = await ls_client.tasks.create(
            data={"image": image_url},
            project=PROJECT_ID
        )

        await ls_client.predictions.create(
            task=task.id,
            model_version="v1-auto",
            result=results_list
        )

        # -------------------- FETCH FINAL LABELS --------------------
        # Wait for LS to complete annotation (if manual correction is done)
        # For demo, we fetch existing annotation directly
        annos = await ls_client.annotations.list(task=task.id) # type: ignore
        yolo_label_path = LABELS_DIR / f"{Path(image_path).stem}.txt"
        LABELS_DIR.mkdir(parents=True, exist_ok=True)

        with open(yolo_label_path, "w") as f:
            for a in annos:
                for r in a.result:
                    if r["type"] == "rectanglelabels":
                        label = r["value"]["rectanglelabels"][0]
                        x = (r["value"]["x"] + r["value"]["width"] / 2) / 100
                        y = (r["value"]["y"] + r["value"]["height"] / 2) / 100
                        w = r["value"]["width"] / 100
                        h = r["value"]["height"] / 100
                        cls_id = classes.index(label)
                        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

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
                message=f"✅ {len(detections)} boxes uploaded to Label Studio & retraining started",
                mode="auto",
                image=str(dataset_img),
                label_file=str(yolo_label_path),
                label_name=label_name,
                classes=classes
            )
        }

    except Exception as e:
        processing_tasks[task_id] = {"status": "error", "error": str(e)}
