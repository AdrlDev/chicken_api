import os
import shutil
import threading
from ultralytics import YOLO  # type: ignore
from app.utils import BASE_DIR, DATASET_DIR, YOLO_WEIGHTS
from app.ws_manager import ws_manager
import asyncio
import json
import re

# Thread lock for safe YOLO reloading
reload_lock = threading.Lock()

class WSLogger:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        # Each WSLogger gets its own event loop for threads
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def __call__(self, info):
        msg_str = str(info)
        send_obj = {"log": msg_str, "progress": None}

        # Detect epoch progress
        match = re.search(r"epoch (\d+)/(\d+)", msg_str.lower())
        if match:
            epoch = int(match.group(1))
            total = int(match.group(2))
            progress = int(epoch / total * 100)
            send_obj["progress"] = progress

        # Send JSON via WebSocket
        json_msg = json.dumps(send_obj)
        try:
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast(json_msg), self.loop)
        except RuntimeError:
            asyncio.run(ws_manager.broadcast(json_msg))

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

    # Merge new images
    merged = safe_merge_new_images(images_dir, labels_dir, val_ratio)
    if merged == 0:
        if not os.listdir(os.path.join(images_dir, "train")) and not os.listdir(os.path.join(images_dir, "val")):
            raise RuntimeError("❌ No training data found! Both train/val are empty.")

    data_yaml_path = update_data_yaml(dataset_dir)

    # Device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Training on device: {device}")

    # Disable multiprocessing inside thread
    os.environ["YOLO_NO_MULTIPROCESSING"] = "1"

    try:
        # Load model
        if os.path.exists(best_weights_path):
            model = YOLO(best_weights_path)
            model.to(device)
        else:
            model = YOLO(YOLO_WEIGHTS)
            model.to(device)

        # Train with WSLogger callback
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            project=save_dir,
            name=train_name,
            exist_ok=True,
            batch=1,
            callbacks=[WSLogger(total_epochs=epochs)]
        )

        # Save best.pt
        source_best = os.path.join(save_dir, train_name, "weights", "best.pt")
        if os.path.exists(source_best):
            shutil.copy(source_best, best_weights_path)
            print(f"🎯 Best weights saved at: {best_weights_path}")
        else:
            raise RuntimeError("❌ Training finished but no best.pt was created!")

        return best_weights_path

    except Exception as e:
        import traceback
        print("❌ YOLO training failed:")
        traceback.print_exc()
        raise e


def _train(dataset_dir=DATASET_DIR, epochs=100, imgsz=640, val_ratio=0.2):
    """Entry point for threaded training"""
    with reload_lock:
        best = train_yolo_autosplit(dataset_dir, epochs, imgsz, val_ratio)

        # Reload model with latest weights
        from app.utils import yolo, get_latest_trained_weights
        new_weights = get_latest_trained_weights()
        if os.path.exists(new_weights):
            yolo = YOLO(new_weights)
            print(f"✅ Model reloaded with new weights: {new_weights}")
        else:
            print("⚠️ No best.pt found after training")
