import os
import random
import shutil
from datetime import datetime
from ultralytics import YOLO  # type: ignore
from app.utils import BASE_DIR, YOLO_WEIGHTS, DATASET_DIR, TRAINED_WEIGHTS
import subprocess
import threading

# Thread lock for safe YOLO reloading
reload_lock = threading.Lock()

def backup_dataset(dataset_dir: str):
    """Create a timestamped backup of the dataset before training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # microseconds for uniqueness
    backup_dir = os.path.join(BASE_DIR, "backups", f"dataset_{timestamp}")
    os.makedirs(os.path.dirname(backup_dir), exist_ok=True)
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(dataset_dir, backup_dir)
    print(f"📦 Dataset backed up to: {backup_dir}")

def train_yolo_autosplit(dataset_dir: str, model_name: str = "yolov8n.pt", epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    """
    Automatically split YOLO dataset (from Label Studio) into train/val, create data.yaml, and train YOLOv8.
    Safe version: does not remove existing files.
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
    split_dirs = [
        os.path.join(images_dir, "train"),
        os.path.join(images_dir, "val"),
        os.path.join(labels_dir, "train"),
        os.path.join(labels_dir, "val"),
    ]
    for d in split_dirs:
        os.makedirs(d, exist_ok=True)

    # Collect all unsplit images (ignore those already inside /train or /val)
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not os.path.isdir(os.path.join(images_dir, f))
    ]

    # Shuffle and split only new ones
    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - val_ratio))
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]

    # ✅ Copy (not move) to keep dataset intact
    for subset, files in [("train", train_images), ("val", val_images)]:
        for img_file in files:
            base_name = os.path.splitext(img_file)[0]
            src_img = os.path.join(images_dir, img_file)
            src_label = os.path.join(labels_dir, f"{base_name}.txt")
            dst_img = os.path.join(images_dir, subset, img_file)
            dst_label = os.path.join(labels_dir, subset, f"{base_name}.txt")

            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

    # 🧩 Ensure there’s at least one validation image
    val_img_dir = os.path.join(images_dir, "val")
    val_lbl_dir = os.path.join(labels_dir, "val")
    train_img_dir = os.path.join(images_dir, "train")
    train_lbl_dir = os.path.join(labels_dir, "train")

    if len(os.listdir(val_img_dir)) == 0:
        print("⚠️ No validation images found. Copying 1 sample from train...")
        train_imgs = [f for f in os.listdir(train_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if train_imgs:
            img = train_imgs[0]
            base_name = os.path.splitext(img)[0]
            shutil.copy2(os.path.join(train_img_dir, img), os.path.join(val_img_dir, img))
            lbl_src = os.path.join(train_lbl_dir, f"{base_name}.txt")
            lbl_dst = os.path.join(val_lbl_dir, f"{base_name}.txt")
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, lbl_dst)

    # Create data.yaml
    yaml_content = f"""train: {os.path.join(images_dir, 'train')}
    val: {os.path.join(images_dir, 'val')}

    nc: {nc}
    names: {class_names}
    """
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ Dataset split: {len(train_images)} train, {len(val_images)} val")
    print(f"✅ Created data.yaml with {nc} classes: {class_names}")

    # Train YOLO
    model = YOLO(model_name)
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz
    )

    print("🎯 Training complete! Check 'runs/detect/train/' for results.")


def _train():
    print("🚀 Starting YOLO training...")
    train_yolo_autosplit(
        dataset_dir=DATASET_DIR,
        model_name=YOLO_WEIGHTS,
        epochs=100,
        imgsz=640,
        val_ratio=0.2
    )

    # Safely reload updated model
    with reload_lock:
        import app.utils as utils
        new_weights = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
        if os.path.exists(new_weights):
            utils.yolo = YOLO(new_weights)
            print(f"✅ Model reloaded with new weights: {new_weights}")


def _train_auto(timestamp: str):
    """Run short fine-tuning on new auto-labeled images."""
    def _run():
        backup_dataset(DATASET_DIR)
        subprocess.run([
            "yolo",
            "detect",
            "train",
            f"model={TRAINED_WEIGHTS if os.path.exists(TRAINED_WEIGHTS) else YOLO_WEIGHTS}",
            f"data={os.path.join(DATASET_DIR, 'data.yaml')}",
            "epochs=5",
            "imgsz=640",
            f"project={os.path.join(BASE_DIR, 'runs/auto_train')}",
            f"name=train_{timestamp}",
            "--exist-ok"
        ], check=True)

        # Reload updated weights safely
        with reload_lock:
            import app.utils as utils
            new_weights = os.path.join(BASE_DIR, "runs", "auto_train", f"train_{timestamp}", "weights", "best.pt")
            if os.path.exists(new_weights):
                utils.yolo = YOLO(new_weights)
                print(f"✅ Model reloaded after auto-training: {new_weights}")

    threading.Thread(target=_run, daemon=True).start()
