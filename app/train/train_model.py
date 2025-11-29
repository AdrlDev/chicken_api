# # app/train_model.py
# This module handles YOLOv8 training, dataset management, and WebSocket communication.

import os
import shutil
import threading
import json
import random
import asyncio
from ultralytics import YOLO  # type: ignore
from app.utils.utils import yolo, get_latest_trained_weights
from app.utils.config import DATASET_DIR, YOLO_WEIGHTS, BASE_DIR
from app.utils.ws_manager import ws_manager
import torch

# -------------------
# Device & locks
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Training on device: {device}")

reload_lock = threading.Lock()
MAIN_LOOP = asyncio.get_event_loop()  # capture main loop for WS

# -------------------
# Helper: safe float conversion
# -------------------
def safe_float(v):
    try:
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except Exception:
        return None

# -------------------
# Callbacks
# -------------------
def send_ws_event(event: dict):
    """Thread-safe broadcast to WebSocket."""
    try:
        msg = json.dumps(event)
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(msg), MAIN_LOOP)
    except Exception as e:
        print("❌ Failed to send WS event:", e)

def on_train_batch_end_callback(trainer):
    info = {
        "event": "batch_end",
        "epoch": int(getattr(trainer, "epoch", 0)),
        "batch": int(getattr(trainer, "batch", 0)),
        "total_batches": int(getattr(trainer, "num_batches", 1)),
        "progress": round(
            int(getattr(trainer, "batch", 0)) / max(int(getattr(trainer, "num_batches", 1)), 1) * 100
        ),
        "loss": safe_float(getattr(trainer, "loss", None))
    }
    send_ws_event(info)

def on_epoch_end_callback(trainer):
    info = {
        "event": "epoch_end",
        "epoch": int(getattr(trainer, "epoch", 0)),
        "total_epochs": int(getattr(trainer, "epochs", 0)),
        "best_fitness": safe_float(getattr(trainer, "best_fitness", None)),
        "metrics": {k: safe_float(v) for k, v in getattr(trainer, "metrics", {}).items()},
    }
    send_ws_event(info)

def on_model_save_callback(trainer):
    info = {
        "event": "model_saved",
        "best_fitness": safe_float(getattr(trainer, "best_fitness", None)),
        "total_loss": safe_float(getattr(trainer, "losses", [None])[-1] if getattr(trainer, "losses", None) else None),
        "metrics": {k: safe_float(v) for k, v in getattr(trainer, "metrics", {}).items()},
        "loss_names": list(getattr(trainer, "loss_names", []))
    }
    send_ws_event(info)

# -------------------
# Dataset utilities
# -------------------
def safe_merge_new_images(images_dir: str, labels_dir: str, val_ratio: float = 0.2):
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, subset), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, subset), exist_ok=True)

    new_images = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not os.path.isdir(os.path.join(images_dir, f))
    ]

    if not new_images:
        print("ℹ️ No new images found to merge.")
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast("ℹ️ No new images found to merge."),
            MAIN_LOOP
        )
        return 0

    random.shuffle(new_images)
    split_idx = int(len(new_images) * (1 - val_ratio))
    train_imgs = new_images[:split_idx] or new_images
    val_imgs = new_images[split_idx:] or new_images[-1:]

    moved_count = 0
    for subset, files in [("train", train_imgs), ("val", val_imgs)]:
        for img_file in files:
            base = os.path.splitext(img_file)[0]
            src_img = os.path.join(images_dir, img_file)
            src_lbl = os.path.join(labels_dir, f"{base}.txt")
            dst_img = os.path.join(images_dir, subset, img_file)
            dst_lbl = os.path.join(labels_dir, subset, f"{base}.txt")

            if os.path.exists(dst_img) or not os.path.exists(src_img):
                continue

            try:
                shutil.move(src_img, dst_img)
                if os.path.exists(src_lbl):
                    shutil.move(src_lbl, dst_lbl)
                moved_count += 1
            except Exception as move_err:
                print(f"⚠️ Failed to move {img_file}: {move_err}")

    print(f"📥 Merged {moved_count} new images into dataset.")
    asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast(f"📥 Merged {moved_count} new images into dataset."),
            MAIN_LOOP
        )
    return moved_count

def update_data_yaml(dataset_dir: str):
    images_dir = os.path.join(dataset_dir, "images")
    classes_path = os.path.join(dataset_dir, "classes.txt")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")

    class_names = []
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]

    yaml_content = (
        f"train: {os.path.abspath(os.path.join(images_dir, 'train'))}\n"
        f"val: {os.path.abspath(os.path.join(images_dir, 'val'))}\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    return data_yaml_path

# -------------------
# Training
# -------------------
def train_yolo_autosplit(dataset_dir: str, epochs: int = 100, imgsz: int = 416, val_ratio: float = 0.2):
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    save_dir = os.path.join(BASE_DIR, "runs", "detect")
    train_dir = os.path.join(save_dir, "train")
    weights_dir = os.path.join(train_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # -------------------------------
    # 1. Merge new uploaded images
    # -------------------------------
    safe_merge_new_images(images_dir, labels_dir, val_ratio)

    # -------------------------------
    # 2. ALWAYS update data.yaml
    # -------------------------------
    data_yaml_path = update_data_yaml(dataset_dir)

    # -------------------------------
    # 3. Select correct weights
    # -------------------------------
    weights_to_use = get_latest_trained_weights()
    print(f"🔧 Loading model weights: {weights_to_use}")

    model = YOLO(weights_to_use)
    model.to(device)

    # -------------------------------
    # 4. Attach callbacks
    # -------------------------------
    model.add_callback("on_model_save", on_model_save_callback)
    model.add_callback("on_epoch_end", on_epoch_end_callback)
    model.add_callback("on_train_batch_end", on_train_batch_end_callback)

    # -------------------------------
    # 5. Train model
    # -------------------------------
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        project=save_dir,
        name="train",
        exist_ok=True,
        cache="ram"
    )

    # -------------------------------
    # 6. Save best.pt into a stable location
    # -------------------------------
    final_best = os.path.join(train_dir, "weights", "best.pt")

    if os.path.exists(final_best):
        print(f"🎯 Best weights updated: {final_best}")
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast(f"🎯 Best weights updated: {final_best}"),
            MAIN_LOOP
        )
    else:
        print("⚠️ WARNING: best.pt NOT FOUND — fallback to base model.")
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast("⚠️ WARNING: best.pt NOT FOUND — fallback to base model."),
            MAIN_LOOP
        )
        final_best = YOLO_WEIGHTS

    return final_best

# -------------------
# Threaded training
# -------------------
def _train(dataset_dir=str(DATASET_DIR), epochs=100, imgsz=640, val_ratio=0.2):
    """Threaded YOLOv8 training with WS stream"""
    with reload_lock:
        latest_weights = train_yolo_autosplit(dataset_dir, epochs, imgsz, val_ratio)

        # Reload model globally
        global yolo
        yolo = YOLO(latest_weights)
        yolo.to(device)

        msg = f"🔄 Model reloaded with latest trained weights: {latest_weights}"
        print(msg)

        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast(msg),
            MAIN_LOOP
        )