import os
import random
import shutil
from ultralytics import YOLO # type: ignore

def train_yolo_autosplit(dataset_dir: str, model_name: str = "yolov8n.pt", epochs: int = 50, imgsz: int = 640, val_ratio: float = 0.2):
    """
    Automatically split YOLO dataset (from Label Studio) into train/val, create data.yaml, and train YOLOv8.

    Args:
        dataset_dir (str): Path to dataset directory containing 'images/', 'labels/', 'classes.txt'.
        model_name (str): YOLO model (e.g. 'yolov8n.pt').
        epochs (int): Number of training epochs.
        imgsz (int): Image size.
        val_ratio (float): Portion of data to use for validation (default 0.2).
    """

    # Paths
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    classes_path = os.path.join(dataset_dir, "classes.txt")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")

    # Check dataset
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

    # Collect all images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - val_ratio))
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]

    # Move images + labels into split folders
    for subset, files in [("train", train_images), ("val", val_images)]:
        for img_file in files:
            base_name = os.path.splitext(img_file)[0]
            src_img = os.path.join(images_dir, img_file)
            src_label = os.path.join(labels_dir, f"{base_name}.txt")
            
            dst_img = os.path.join(images_dir, subset, img_file)
            dst_label = os.path.join(labels_dir, subset, f"{base_name}.txt")

            # Move files (skip if already moved)
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)

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