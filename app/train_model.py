import os
import random
import shutil
import threading
import subprocess
from datetime import datetime
from ultralytics import YOLO  # type: ignore
from app.utils import BASE_DIR, YOLO_WEIGHTS, DATASET_DIR, TRAINED_WEIGHTS, IMAGES_DIR, LABELS_DIR

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
    Automatically split YOLO dataset (from Label Studio) into train/val,
    create data.yaml, and train YOLOv8.
    """

    # --- Backup before training ---
    backup_dataset(dataset_dir)

    # Paths
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    classes_path = os.path.join(dataset_dir, "classes.txt")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError("Missing 'images/' or 'labels/' directory in dataset.")

    # Read class names
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    nc = len(class_names)

    # Prepare split directories
    for sub in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, sub), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, sub), exist_ok=True)

    # Collect all unsplit images (ignore ones already in /train or /val)
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not os.path.isdir(os.path.join(images_dir, f))
    ]

    if not image_files:
        raise RuntimeError("❌ No images found in dataset/images")

    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - val_ratio))
    train_images = image_files[:split_idx] or image_files
    val_images = image_files[split_idx:] or image_files[-1:]

    # Move files to train/val
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

    # Create a clean YAML file (⚠️ Must have no extra indentation)
    yaml_content = (
        f"train: {os.path.join(images_dir, 'train')}\n"
        f"val: {os.path.join(images_dir, 'val')}\n\n"
        f"nc: {nc}\n"
        f"names: {class_names}\n"
    )

    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ Created data.yaml with {nc} classes: {class_names}")

    # --- Train YOLO ---
    save_dir = os.path.join(BASE_DIR, "runs", "detect")
    os.makedirs(save_dir, exist_ok=True)

    try:
        model = YOLO(model_name)
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            project=save_dir,
            name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            exist_ok=True
        )
        print("🎯 Training complete! Check runs/detect/ for results.")
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

def _train_auto(epochs: int = 5, imgsz: int = 640):
    """
    Incrementally fine-tune YOLO:
    - Uses latest best.pt weights
    - Merges old + new images/labels
    - Updates TRAINED_WEIGHTS in place
    """
    def _run():
        try:
            # 1️⃣ Merge new data into main train folders
            new_images_dir = os.path.join(IMAGES_DIR, "new")
            new_labels_dir = os.path.join(LABELS_DIR, "new")
            if os.path.exists(new_images_dir):
                for f in os.listdir(new_images_dir):
                    shutil.move(os.path.join(new_images_dir, f), IMAGES_DIR)
                shutil.rmtree(new_images_dir)
            if os.path.exists(new_labels_dir):
                for f in os.listdir(new_labels_dir):
                    shutil.move(os.path.join(new_labels_dir, f), LABELS_DIR)
                shutil.rmtree(new_labels_dir)

            # 2️⃣ Prepare data.yaml
            images_dir = os.path.join(DATASET_DIR, "images")
            classes_path = os.path.join(DATASET_DIR, "classes.txt")
            data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")

            # Load classes
            with open(classes_path, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]

            nc = len(class_names)

            # YAML content
            yaml_content = f"""
            train: {os.path.join(images_dir, 'train')}
            val: {os.path.join(images_dir, 'val')}
            nc: {nc}
            names: {class_names}
            """
            with open(data_yaml_path, "w") as f:
                f.write(yaml_content.strip())

            # 3️⃣ Determine latest weights
            latest_weights = TRAINED_WEIGHTS if os.path.exists(TRAINED_WEIGHTS) else os.path.join(BASE_DIR, "assets", "yolov8n.pt")
            print(f"🔁 Auto-train starting from: {latest_weights}")

            # 4️⃣ Train YOLO
            model = YOLO(latest_weights)
            model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                project=os.path.join(BASE_DIR, "runs", "detect"),
                name="train",
                exist_ok=True
            )

            # 5️⃣ Update TRAINED_WEIGHTS
            best_path = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
            if os.path.exists(best_path):
                from app import utils
                with reload_lock:
                    utils.TRAINED_WEIGHTS = best_path
                    utils.yolo = YOLO(best_path)
                    print(f"✅ Model updated with new best.pt: {best_path}")
            else:
                print("⚠️ Auto-train finished but no best.pt found.")

        except Exception as e:
            import traceback
            print("❌ Auto-train failed:")
            traceback.print_exc()

    threading.Thread(target=_run, daemon=True).start()