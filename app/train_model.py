# train_model.py

import os
import random
import shutil
import threading
import subprocess
from datetime import datetime
from ultralytics import YOLO  # type: ignore
from app.utils import BASE_DIR, YOLO_WEIGHTS, DATASET_DIR

# Thread lock for safe YOLO reloading
reload_lock = threading.Lock()


def backup_dataset(dataset_dir: str):
    """Create a timestamped backup of the dataset before training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # microseconds for uniqueness
    backup_dir = os.path.join(BASE_DIR, "backups", f"dataset_{timestamp}")
    os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
    shutil.copytree(dataset_dir, backup_dir, dirs_exist_ok=True)
    print(f"📦 Dataset backed up to: {backup_dir}")

def safe_merge_new_images(images_dir: str, labels_dir: str, val_ratio: float = 0.2):
    """
    Merge only NEW images (not already in train/val) into dataset.
    Keeps existing data and splits new ones into train/val.
    """
    # Prepare folders
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, subset), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, subset), exist_ok=True)

    # Find new (unassigned) images
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

    # Move new images into proper folders
    moved_count = 0
    for subset, files in [("train", train_imgs), ("val", val_imgs)]:
        for img_file in files:
            base = os.path.splitext(img_file)[0]
            src_img = os.path.join(images_dir, img_file)
            src_lbl = os.path.join(labels_dir, f"{base}.txt")
            dst_img = os.path.join(images_dir, subset, img_file)
            dst_lbl = os.path.join(labels_dir, subset, f"{base}.txt")

            # Skip if already exists in train/val
            if os.path.exists(dst_img):
                continue

            shutil.move(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
            moved_count += 1

    print(f"📥 Merged {moved_count} new images into dataset.")
    return moved_count


def train_yolo_autosplit(dataset_dir: str, model_name: str = "yolov8n.pt",
                         epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    """
    Train YOLO on the dataset, updating the existing 'train' folder with new data.
    """

    # Paths
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    save_dir = os.path.join(BASE_DIR, "runs", "detect")
    train_name = "train"  # always the same folder
    train_path = os.path.join(save_dir, train_name)

    # --- Merge only new data ---
    merged = safe_merge_new_images(images_dir, labels_dir, val_ratio)
    if merged == 0:
        train_dir = os.path.join(images_dir, "train")
        val_dir = os.path.join(images_dir, "val")
        if not os.listdir(train_dir) and not os.listdir(val_dir):
            raise RuntimeError("❌ No training data found! Both train/val are empty.")

    update_data_yaml(images_dir)

    # --- Train YOLO ---
    os.makedirs(save_dir, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Training on device: {device}")

    try:
        model = YOLO(model_name)
        model.to(device)
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            project=save_dir,
            name=train_name,   # same folder every time
            exist_ok=True,     # overwrite existing folder
            save=True,
            save_period=1
        )

        best_path = os.path.join(train_path, "weights", "best.pt")
        if os.path.exists(best_path):
            print(f"🎯 Best weights updated at: {best_path}")
        else:
            print("⚠️ No best.pt found after training!")

        return best_path if os.path.exists(best_path) else None

    except Exception as e:
        import traceback
        print("❌ YOLO training failed:")
        traceback.print_exc()
        raise e

def _train():
    """Manual training endpoint logic"""
    print("🚀 Starting YOLO training...")
    train_yolo_autosplit(
        dataset_dir=DATASET_DIR,
        model_name=YOLO_WEIGHTS,
        epochs=100,
        imgsz=640,
        val_ratio=0.2
    )

    # Reload trained model weights
    with reload_lock:
        import app.utils as utils
        new_weights = utils.get_latest_trained_weights()
        if os.path.exists(new_weights):
            utils.yolo = YOLO(new_weights)
            print(f"✅ Model reloaded with new weights: {new_weights}")
        else:
            print("⚠️ No best.pt found after training.")


def update_data_yaml(dataset_dir: str):
    """
    Regenerate data.yaml automatically based on dataset/images/train & val,
    and current classes.txt.
    """
    images_dir = os.path.join(dataset_dir, "images")
    classes_path = os.path.join(dataset_dir, "classes.txt")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")

    # Read classes
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        class_names = []

    # Create YAML content
    yaml_content = (
        f"train: {os.path.abspath(os.path.join(images_dir, 'train'))}\n"
        f"val: {os.path.abspath(os.path.join(images_dir, 'val'))}\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    # Write to file
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    return data_yaml_path

def _train_auto(epochs: int = 5, imgsz: int = 640, auto_label: bool = True):
    """
    Incrementally fine-tune YOLO using auto-split dataset:
    - Adds new images to the existing train/val
    - Always updates 'runs/detect/train/weights/best.pt'
    - Auto-labels new images if enabled
    """
    import asyncio

    async def _run():
        try:
            print("\n🚀 Starting auto-training process...")
            print("=" * 50)

            # --- Auto-label new images ---
            if auto_label:
                try:
                    from app.auto_labeler import ChickenAutoLabeler
                    print("\n🏷️ Running auto-labeling on new images...")

                    label_studio_url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
                    api_key = os.getenv("LABEL_STUDIO_API_KEY")

                    if not api_key:
                        print("⚠️ LABEL_STUDIO_API_KEY not set, skipping auto-labeling")
                    else:
                        labeler = ChickenAutoLabeler(label_studio_url, api_key)
                        results = await labeler.predict_and_label()
                        print(f"✅ Auto-labeled {results['labeled']} images")
                        if results['errors']:
                            print(f"⚠️ Encountered {len(results['errors'])} errors during labeling")
                except Exception as e:
                    print(f"⚠️ Auto-labeling failed: {str(e)}\nContinuing with training...")

            # --- Train YOLO ---
            from app.utils import get_latest_trained_weights
            latest_weights = get_latest_trained_weights()

            print("🎬 Starting incremental training on 'train' folder...")
            best_path = train_yolo_autosplit(
                dataset_dir=DATASET_DIR,
                model_name=latest_weights,
                epochs=epochs,
                imgsz=imgsz,
                val_ratio=0.2
            )

            # --- Reload YOLO model ---
            if best_path and os.path.exists(best_path):
                from app import utils
                with reload_lock:
                    utils.yolo = YOLO(best_path)
                    print(f"✅ Model reloaded with updated weights: {best_path}")
            else:
                print("⚠️ No best.pt found after incremental training!")

        except Exception as e:
            import traceback
            print("❌ Auto-train failed:")
            traceback.print_exc()

    threading.Thread(target=lambda: asyncio.run(_run()), daemon=True).start()