#auto_labeler.py

import os
import json
import requests
from typing import List, Dict, Any
from label_studio_sdk import Client
from ultralytics import YOLO
import numpy as np
import cv2
from app.utils import get_latest_trained_weights

class ChickenAutoLabeler:
    def __init__(
        self,
        label_studio_url: str = "http://localhost:8080",
        api_key: str = None # type: ignore
    ):
        """Initialize connection to Label Studio and load YOLO model"""
        self.ls = Client(url=label_studio_url, api_key=api_key)
        self.model = YOLO(get_latest_trained_weights())
        
        # Get project info
        self.project = self.ls.get_projects()[0]  # type: ignore # Assuming single project
        print(f"📊 Connected to project: {self.project.title}")
        
        # Load class mapping
        self.class_names = self._load_class_names()
        print(f"🏷️ Loaded {len(self.class_names)} classes")

    def _load_class_names(self) -> List[str]:
        """Load class names from classes.txt"""
        classes_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "classes.txt")
        with open(classes_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _convert_yolo_to_ls_format(
        self,
        predictions: List[Dict],
        image_width: int,
        image_height: int
    ) -> List[Dict]:
        """Convert YOLO predictions to Label Studio format"""
        results = []
        
        for pred in predictions:
            boxes = pred.boxes # type: ignore
            for box in boxes:
                # Get normalized coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                cls = int(box.cls)
                
                # Convert to absolute coordinates
                x = x1 / image_width * 100
                y = y1 / image_height * 100
                width = (x2 - x1) / image_width * 100
                height = (y2 - y1) / image_height * 100
                
                # Create Label Studio compatible annotation
                results.append({
                    "type": "rectanglelabels",
                    "value": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "rotation": 0,
                        "rectanglelabels": [self.class_names[cls]]
                    },
                    "score": conf,
                    "origin": "prediction"
                })
        
        return results

    async def predict_and_label(self, task_ids: List[int] = None) -> Dict[str, Any]: # type: ignore
        """
        Run YOLO predictions on Label Studio tasks and create annotations
        Args:
            task_ids: List of task IDs to process. If None, process all tasks
        """
        if task_ids is None:
            # Get all tasks without annotations
            tasks = self.project.get_tasks()
            task_ids = [t['id'] for t in tasks if not t.get('annotations')]
        
        results = {
            "processed": 0,
            "labeled": 0,
            "errors": []
        }

        for task_id in task_ids:
            try:
                # Get task data
                task = self.project.get_task(task_id)
                if not task:
                    results["errors"].append(f"Task {task_id} not found")
                    continue

                # Download image
                img_url = task["data"]["image"]
                response = requests.get(img_url)
                if response.status_code != 200:
                    results["errors"].append(f"Failed to download image for task {task_id}")
                    continue

                # Convert to numpy array
                nparr = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                h, w = img.shape[:2] # type: ignore

                # Run YOLO prediction
                predictions = self.model.predict(img, conf=0.25) # type: ignore
                
                # Convert predictions to Label Studio format
                annotations = self._convert_yolo_to_ls_format(predictions, w, h) # type: ignore
                
                if annotations:
                    # Create annotation in Label Studio
                    self.project.create_annotation(
                        task_id,
                        result=annotations,
                        model_version=self.model.model.yaml_file # type: ignore
                    )
                    results["labeled"] += 1
                
                results["processed"] += 1
                
                print(f"✅ Processed task {task_id}: {len(annotations)} detections")
                
            except Exception as e:
                error_msg = f"Error processing task {task_id}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"❌ {error_msg}")
                continue
                
        return results

    def train_and_update(self):
        """
        Export annotations, train model, and update predictions
        """
        # Export annotations to YOLO format
        export_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
        self.project.export_tasks(export_dir, format="YOLO")
        
        # Trigger training
        from app.train_model import _train_auto
        _train_auto(epochs=10)  # Quick fine-tuning
        
        # Update model with new weights
        self.model = YOLO(get_latest_trained_weights())
        print("✅ Model updated with new weights")