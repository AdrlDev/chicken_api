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
    - Handles GPU/CPU training efficiently
    """
    def _run():
        try:
            from app.utils import ModelManager, DEVICE
            print("🚀 Starting auto-training process...")

            # 1️⃣ Merge new data into main train folders
            try:
                # Create train directories if they don't exist
                for d in [os.path.join(IMAGES_DIR, "train"), os.path.join(LABELS_DIR, "train")]:
                    os.makedirs(d, exist_ok=True)

                new_images_dir = os.path.join(IMAGES_DIR, "new")
                new_labels_dir = os.path.join(LABELS_DIR, "new")
                
                # Move new data to train folders
                if os.path.exists(new_images_dir):
                    for f in os.listdir(new_images_dir):
                        src = os.path.join(new_images_dir, f)
                        dst = os.path.join(IMAGES_DIR, "train", f)
                        if os.path.exists(src):
                            shutil.move(src, dst)
                    if os.path.exists(new_images_dir):
                        shutil.rmtree(new_images_dir)
                
                if os.path.exists(new_labels_dir):
                    for f in os.listdir(new_labels_dir):
                        src = os.path.join(new_labels_dir, f)
                        dst = os.path.join(LABELS_DIR, "train", f)
                        if os.path.exists(src):
                            shutil.move(src, dst)
                    if os.path.exists(new_labels_dir):
                        shutil.rmtree(new_labels_dir)
                        
                print("✅ Successfully merged new data into training folders")
            except Exception as e:
                print(f"❌ Error merging data: {str(e)}")
                raise

            # 2️⃣ Prepare data.yaml
            images_dir = os.path.join(DATASET_DIR, "images")
            classes_path = os.path.join(DATASET_DIR, "classes.txt")
            data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")

            # Load classes
            with open(classes_path, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]

            nc = len(class_names)

            # YAML content with proper quoting
            yaml_content = (
                f"train: {os.path.join(images_dir, 'train')}\n"
                f"val: {os.path.join(images_dir, 'val')}\n"
                f"nc: {nc}\n"
                f"names: {class_names}\n"
            )
            with open(data_yaml_path, "w") as f:
                f.write(yaml_content)

            # 3️⃣ Get latest trained weights to continue training
            from app.utils import get_latest_trained_weights
            latest_weights = get_latest_trained_weights()
            print(f"🔄 Continuing training from weights: {latest_weights}")
            
            # Create new model instance with latest weights
            model = YOLO(latest_weights)
            print(f"🖥️ Training on {DEVICE.upper()}")

            # 4️⃣ Train YOLO with optimized settings for CPU/GPU
            # Create a timestamped run folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"train_{timestamp}"
            
            train_args = {
                "data": data_yaml_path,
                "epochs": epochs,
                "imgsz": imgsz,
                "project": os.path.join(BASE_DIR, "runs", "detect"),
                "name": run_name,
                "exist_ok": True,
                "device": DEVICE,
                "batch": 8 if DEVICE == "cuda" else 1,  # Smaller batch size for CPU
                "workers": 4 if DEVICE == "cuda" else 0,  # Disable workers on CPU
                "verbose": True,
                "resume": False,  # Don't resume interrupted training
                "pretrained": True,  # Use pretrained weights
                "weights": latest_weights  # Continue from latest weights
            }

            print(f"📊 Training args: {train_args}")
            model.train(**train_args)

            # 5️⃣ Update TRAINED_WEIGHTS
            best_path = os.path.join(BASE_DIR, "runs", "detect", run_name, "weights", "best.pt")
            if os.path.exists(best_path):
                from app import utils
                with reload_lock:
                    utils.TRAINED_WEIGHTS = best_path
                    utils.yolo = YOLO(best_path)
                    print(f"✅ Model updated with new best.pt: {best_path}")
                    print(f"🎯 Training results saved in: {os.path.join(BASE_DIR, 'runs', 'detect', run_name)}")
            else:
                print(f"⚠️ Auto-train finished but no best.pt found in {best_path}")
                print("⚠️ Training might have failed - check the logs above for errors")

        except Exception as e:
            import traceback
            print("❌ Auto-train failed:")
            traceback.print_exc()

    threading.Thread(target=_run, daemon=True).start()