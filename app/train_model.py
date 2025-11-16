import os
import shutil
import threading
from ultralytics import YOLO  # type: ignore
from app.utils import BASE_DIR, DATASET_DIR, YOLO_WEIGHTS, yolo, get_latest_trained_weights
from app.ws_manager import ws_manager
import json
import asyncio

# Thread lock for safe YOLO reloading
reload_lock = threading.Lock()

# Device
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Training on device: {device}")

def on_train_batch_end_callback(trainer):
    try:
        # Calculate progress
        current_batch = int(trainer.batch) if hasattr(trainer, "batch") else 0
        total_batches = int(trainer.num_batches) if hasattr(trainer, "num_batches") else 1
        progress = round((current_batch / total_batches) * 100)

        # JSON-serializable info
        info = {
            "event": "batch_end",
            "epoch": int(trainer.epoch) if hasattr(trainer, "epoch") else None,
            "batch": current_batch,
            "total_batches": total_batches,
            "progress": progress,
            "loss": float(trainer.loss) if hasattr(trainer, "loss") else None
        }

        msg = json.dumps(info)

        # Send via WebSocket
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(msg), loop)

    except Exception as e:
        print("❌ Failed to send batch info:", e)

def on_epoch_end_callback(trainer):
    try:
        # JSON-serializable info
        info = {
            "event": "epoch_end",
            "epoch": int(trainer.epoch) if hasattr(trainer, "epoch") else None,
            "total_epochs": int(trainer.epochs) if hasattr(trainer, "epochs") else None,
            "best_fitness": float(trainer.best_fitness) if hasattr(trainer, "best_fitness") else None,
            "metrics": {k: float(v) for k, v in trainer.metrics.items()} if hasattr(trainer, "metrics") else {},
        }
        msg = json.dumps(info)

        # Send via WebSocket
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(msg), loop)
    except Exception as e:
        print("❌ Failed to send epoch info:", e)


def on_model_save_callback(trainer):
    try:
        # Only extract JSON-serializable info
        info = {
            "event": "model_saved",
            "best_fitness": float(trainer.best_fitness) if hasattr(trainer, "best_fitness") else None,
            "total_loss": float(trainer.losses[-1]) if hasattr(trainer, "losses") and len(trainer.losses) > 0 else None,
            "metrics": {k: float(v) for k, v in trainer.metrics.items()} if hasattr(trainer, "metrics") else {},
            "loss_names": list(trainer.loss_names) if hasattr(trainer, "loss_names") else []
        }

        msg = json.dumps(info)

        # Send via WebSocket (thread-safe)
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(msg), loop)

    except Exception as e:
        print("❌ Failed to send model save info:", e)

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
        return 0

    import random
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
    return moved_count


def update_data_yaml(dataset_dir: str):
    images_dir = os.path.join(dataset_dir, "images")
    classes_path = os.path.join(dataset_dir, "classes.txt")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")

    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        class_names = []

    yaml_content = (
        f"train: {os.path.abspath(os.path.join(images_dir, 'train'))}\n"
        f"val: {os.path.abspath(os.path.join(images_dir, 'val'))}\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    return data_yaml_path


def train_yolo_autosplit(dataset_dir: str, epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    save_dir = os.path.join(BASE_DIR, "runs", "detect")
    train_name = "train"
    train_path = os.path.join(save_dir, train_name)
    weights_dir = os.path.join(train_path, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_dir, "best.pt")
    fresh_yaml_path = os.path.join(dataset_dir, "yolov8n.yaml")

    # Merge new images
    merged = safe_merge_new_images(images_dir, labels_dir, val_ratio)

    # -------------------
    # Fallback: create empty dataset if needed
    # -------------------
    if merged == 0 and not os.listdir(os.path.join(images_dir, "train")) and not os.listdir(os.path.join(images_dir, "val")):
        print("ℹ️ Dataset is empty. Using base YOLOv8n pretrain and creating empty YAML.")
        os.makedirs(os.path.join(images_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(images_dir, "val"), exist_ok=True)

        # Create empty YAML
        if not os.path.exists(fresh_yaml_path):
            yaml_content = (
                f"train: {os.path.join(images_dir, 'train')}\n"
                f"val: {os.path.join(images_dir, 'val')}\n"
                f"nc: 0\n"
                f"names: []\n"
            )
            with open(fresh_yaml_path, "w") as f:
                f.write(yaml_content)
        data_yaml_path = fresh_yaml_path

        # Use base YOLOv8n
        model = YOLO(YOLO_WEIGHTS)
        model.to(device)

    else:
        data_yaml_path = update_data_yaml(dataset_dir)
        model = YOLO(get_latest_trained_weights())
        model.to(device)

    # -------------------
    # Add callback
    # -------------------
    model.add_callback("on_model_save", on_model_save_callback)
    model.add_callback("on_epoch_end", on_epoch_end_callback)
    model.add_callback("on_train_batch_end", on_train_batch_end_callback)

    # -------------------
    # Train
    # -------------------
    print(f"🧠 Training on device: {device}")
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=1,
        project=save_dir,
        name=train_name,
        exist_ok=True
    )

    # -------------------
    # Save best.pt
    # -------------------
    source_best = os.path.join(save_dir, train_name, "weights", "best.pt")
    if os.path.exists(source_best):
        shutil.copy(source_best, best_weights_path)
        print(f"🎯 Best weights saved at: {best_weights_path}")
    else:
        print("⚠️ Training finished but no best.pt was created!")
        best_weights_path = YOLO_WEIGHTS

    return best_weights_path

def stream_train_logs(model, data_yaml_path, epochs=50, imgsz=640, train_name="train"):
    model.add_callback("on_model_save", on_model_save_callback)
    model.add_callback("on_epoch_end", on_epoch_end_callback)

    try:
        print("🧠 Starting YOLOv8 training...")
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=1,
            project="runs/detect",
            name=train_name,
            exist_ok=True
        )
        msg = json.dumps({"event": "training_finished", "message": "Training completed"})
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(msg), loop)

    except Exception as e:
        msg = json.dumps({"event": "training_failed", "error": str(e)})
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast(msg), loop)
        raise e


def _train(dataset_dir=DATASET_DIR, epochs=100, imgsz=640, val_ratio=0.2):
    """Threaded YOLOv8 training with live WS streaming"""
    with reload_lock:
        best_weights = train_yolo_autosplit(dataset_dir, epochs, imgsz, val_ratio)
        data_yaml_path = os.path.join(dataset_dir, "data.yaml")

        # Reload model with latest weights
        global yolo
        yolo = YOLO(best_weights)
        yolo.to(device)
        print(f"✅ Model reloaded with new weights: {best_weights}")

        # Start training in a separate thread to not block server
        t = threading.Thread(
            target=stream_train_logs,
            args=(yolo, data_yaml_path, epochs, imgsz, "train"),
            daemon=True
        )
        t.start()
