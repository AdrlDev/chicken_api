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


def train_yolo_autosplit(dataset_dir: str, model_name: str = "yolov8n.pt",
                         epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    """
    Train YOLO on the dataset, updating the existing 'train' folder with new data.
    """

    # Paths
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    classes_path = os.path.join(dataset_dir, "classes.txt")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    save_dir = os.path.join(BASE_DIR, "runs", "detect")
    train_name = "train"  # always the same folder
    train_path = os.path.join(save_dir, train_name)

    # --- Prepare directories ---
    for sub in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, sub), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, sub), exist_ok=True)

    # --- Split dataset ---
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))
                   and not os.path.isdir(os.path.join(images_dir, f))]

    if not image_files:
        raise RuntimeError("❌ No images found in dataset/images")

    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - val_ratio))
    train_images = image_files[:split_idx] or image_files
    val_images = image_files[split_idx:] or image_files[-1:]

    for subset, files in [("train", train_images), ("val", val_images)]:
        for img_file in files:
            base_name = os.path.splitext(img_file)[0]
            src_img = os.path.join(images_dir, img_file)
            src_lbl = os.path.join(labels_dir, f"{base_name}.txt")
            dst_img = os.path.join(images_dir, subset, img_file)
            dst_lbl = os.path.join(labels_dir, subset, f"{base_name}.txt")
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)

    # --- Create data.yaml ---
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    nc = len(class_names)

    yaml_content = (
        f"train: {os.path.join(images_dir, 'train')}\n"
        f"val: {os.path.join(images_dir, 'val')}\n\n"
        f"nc: {nc}\n"
        f"names: {class_names}\n"
    )
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    # --- Train YOLO ---
    os.makedirs(save_dir, exist_ok=True)

    try:
        model = YOLO(model_name)
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
        f"train: {os.path.join(images_dir, 'train')}\n"
        f"val: {os.path.join(images_dir, 'val')}\n\n"
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

            # --- Merge new images into existing train/val ---
            images_dir = os.path.join(DATASET_DIR, "images")
            labels_dir = os.path.join(DATASET_DIR, "labels")
            for subset in ["train", "val"]:
                os.makedirs(os.path.join(images_dir, subset), exist_ok=True)
                os.makedirs(os.path.join(labels_dir, subset), exist_ok=True)

            # Collect all unassigned images
            new_images = [
                f for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and not os.path.isdir(os.path.join(images_dir, f))
            ]
            if new_images:
                print(f"📥 Found {len(new_images)} new images. Splitting into train/val...")
                import random
                random.shuffle(new_images)
                split_idx = int(len(new_images) * 0.8)
                train_imgs = new_images[:split_idx] or new_images
                val_imgs = new_images[split_idx:] or new_images[-1:]

                for img_file in train_imgs:
                    base_name = os.path.splitext(img_file)[0]
                    src_img = os.path.join(images_dir, img_file)
                    src_lbl = os.path.join(labels_dir, f"{base_name}.txt")
                    dst_img = os.path.join(images_dir, "train", img_file)
                    dst_lbl = os.path.join(labels_dir, "train", f"{base_name}.txt")
                    if os.path.exists(src_img):
                        shutil.move(src_img, dst_img)
                    if os.path.exists(src_lbl):
                        shutil.move(src_lbl, dst_lbl)

                for img_file in val_imgs:
                    base_name = os.path.splitext(img_file)[0]
                    src_img = os.path.join(images_dir, img_file)
                    src_lbl = os.path.join(labels_dir, f"{base_name}.txt")
                    dst_img = os.path.join(images_dir, "val", img_file)
                    dst_lbl = os.path.join(labels_dir, "val", f"{base_name}.txt")
                    if os.path.exists(src_img):
                        shutil.move(src_img, dst_img)
                    if os.path.exists(src_lbl):
                        shutil.move(src_lbl, dst_lbl)

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

    loop = asyncio.new_event_loop()
    threading.Thread(target=lambda: loop.run_until_complete(_run()), daemon=True).start()