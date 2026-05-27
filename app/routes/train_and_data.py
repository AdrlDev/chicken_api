# app/routes/train_and_data.py

import os
import io
import asyncio
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, status
from pydantic import BaseModel
from app.utils.config import IMAGES_DIR, DATASET_DIR
from app.process_image import process_image, processing_tasks, AutoLabelResponse
from app.train.trainer_ws import start_training_thread

router = APIRouter(
    prefix="",
    tags=["Training & Data"],
)

# Define response models (moved here to keep main.py clean)
class UploadResponse(BaseModel):
    message: str
    image_id: str
    status: str = "processing"

class ProcessingStatusResponse(BaseModel):
    status: str
    result: Optional[AutoLabelResponse] = None
    error: Optional[str] = None

# ---------------------------------
# 🐔 AUTO-LABEL ENDPOINT (ASYNC)
# ---------------------------------
@router.post("/auto-label-train", response_model=UploadResponse)
async def auto_label_train(
    file: UploadFile = File(...),
    label_name: str = Form(...)
):
    if not label_name or not label_name.strip():
        raise HTTPException(status_code=400, detail="Label name is required")

    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    image_filename = f"auto_{task_id}.jpg"
    image_path = Path(IMAGES_DIR) / image_filename

    # Read file contents first
    file_contents = await file.read()

    # --- Validate image (separate from save logic) ---
    try:
        img = Image.open(io.BytesIO(file_contents))
        img.verify()  # verify closes the image, so reopen below to check format
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Reopen after verify() to safely check format
    try:
        img2 = Image.open(io.BytesIO(file_contents))
        if img2.format not in ["JPEG", "PNG"]:
            raise HTTPException(status_code=400, detail=f"Only JPEG and PNG supported, got: {img2.format}")
    except HTTPException:
        raise  # ← re-raise validation errors as-is
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image format: {str(e)}")

    # --- Save and process ---
    try:
        with open(image_path, "wb") as f:
            f.write(file_contents)

        processing_tasks[task_id] = {
            "status": "processing",
            "image_path": str(image_path),
            "label_name": label_name
        }

        asyncio.create_task(process_image(
            task_id=task_id,
            image_path=str(image_path),
            label_name=label_name
        ))

        return UploadResponse(
            message="Image uploaded successfully, processing started",
            image_id=task_id
        )

    except Exception as e:
        if image_path.exists():
            os.unlink(image_path)
        raise HTTPException(status_code=500, detail=f"Failed to save/process image: {str(e)}")


@router.get("/auto-label-train/{task_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(task_id: str):
    """Get the status of an image processing task."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    task = processing_tasks[task_id]
    return ProcessingStatusResponse(
        status=task["status"],
        result=task.get("result"),
        error=task.get("error")
    )

# POST endpoint to start training (non-blocking)
@router.post("/train-model")
async def start_training():
    """Start YOLO training in a separate background thread."""
    start_training_thread(dataset_dir=str(DATASET_DIR), epochs=100, imgsz=640, val_ratio=0.2)
    return {"message": "Training started"}