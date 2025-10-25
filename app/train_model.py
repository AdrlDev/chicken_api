import os
import random
import shutil
import threading
import subprocess
from datetime import datetime
from ultralytics import YOLO  # type: ignore
from app.utils import BASE_DIR, YOLO_WEIGHTS, DATASET_DIR, TRAINED_WEIGHTS

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
    Incremental in-place YOLO training:
    - Uses latest trained weights (old best.pt)
    - Includes all old + new dataset images (train + val)
    - Updates same YOLO model weights
    """
    def _run():
        try:
            # 1️⃣ Backup dataset
            backup_dataset(DATASET_DIR)

            # 2️⃣ Regenerate data.yaml
            images_dir = os.path.join(DATASET_DIR, "images")
            classes_path = os.path.join(DATASET_DIR, "classes.txt")
            data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")

            if os.path.exists(classes_path):
                with open(classes_path, "r") as f:
                    class_names = [line.strip() for line in f.readlines() if line.strip()]
            else:
                class_names = []

            yaml_content = (
                f"train: {os.path.join(images_dir, 'train')}\n"
                f"val: {os.path.join(images_dir, 'val')}\n\n"
                f"nc: {len(class_names)}\n"
                f"names: {class_names}\n"
            )

            with open(data_yaml_path, "w") as f:
                f.write(yaml_content)

            # 3️⃣ Determine latest weights
            latest_weights = TRAINED_WEIGHTS
            if not os.path.exists(latest_weights):
                import glob
                trained_list = sorted(
                    glob.glob(os.path.join(BASE_DIR, "runs/detect/**/weights/best.pt"), recursive=True)
                )
                latest_weights = trained_list[-1] if trained_list else YOLO_WEIGHTS

            print(f"🔁 Incremental auto-train starting from weights: {latest_weights}")

            # 4️⃣ Train using YOLO Python API
            model = YOLO(latest_weights)
            # Use same project folder as original training to overwrite best.pt
            project_dir = os.path.join(BASE_DIR, "runs", "detect")
            run_name = "train"  # existing folder will be updated with new weights

            model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                project=project_dir,
                name=run_name,
                exist_ok=True  # overwrite existing run folder
            )

            # 5️⃣ Reload updated model in memory
            best_path = os.path.join(project_dir, run_name, "weights", "best.pt")
            if os.path.exists(best_path):
                from app import utils
                with reload_lock:
                    utils.yolo = YOLO(best_path)
                    utils.TRAINED_WEIGHTS = best_path
                    print(f"✅ Model reloaded after incremental training: {best_path}")
            else:
                print("⚠️ Incremental train finished but no best.pt found.")

        except Exception as e:
            import traceback
            print("❌ Incremental auto-train failed:")
            traceback.print_exc()

    threading.Thread(target=_run, daemon=True).start()