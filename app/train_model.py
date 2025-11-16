# train_model.py

import os
import shutil
import threading
from ultralytics import YOLO  # type: ignore
from app.utils import BASE_DIR, DATASET_DIR

# Thread lock for safe YOLO reloading
reload_lock = threading.Lock()


def safe_merge_new_images(images_dir: str, labels_dir: str, val_ratio: float = 0.2):
    """
    Merge only NEW images (not already in train/val) into dataset.
    Keeps existing data and splits new ones into train/val.
    Skips missing files safely.
    """
    # Prepare folders
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, subset), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, subset), exist_ok=True)

    # Find new (unassigned) images in the root of images_dir
    new_images = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not os.path.isdir(os.path.join(images_dir, f))
    ]

    if not new_images:
        print("ℹ️ No new images found to merge.")
        return 0
    else:
        print(f"📸 Found {len(new_images)} unassigned images to merge.")

    # Shuffle and split
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

            # Skip if destination already exists
            if os.path.exists(dst_img):
                continue

            # Skip if source image is missing
            if not os.path.exists(src_img):
                print(f"⚠️ Skipping missing image: {src_img}")
                continue

            # Move safely
            try:
                shutil.move(src_img, dst_img)
                if os.path.exists(src_lbl):
                    shutil.move(src_lbl, dst_lbl)
                moved_count += 1
            except Exception as move_err:
                print(f"⚠️ Failed to move {img_file}: {move_err}")
                continue

    print(f"📥 Merged {moved_count} new images into dataset.")
    return moved_count


def train_yolo_autosplit(dataset_dir: str,
                         epochs: int = 50,
                         imgsz: int = 640,
                         val_ratio: float = 0.2):
    """
    Train YOLO on the dataset using ONLY uploaded images:
    - If best.pt exists, continue training from it
    - If no best.pt, train fresh on new images and create best.pt
    """

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    save_dir = os.path.join(BASE_DIR, "runs", "detect")
    train_name = "train"
    train_path = os.path.join(save_dir, train_name)
    weights_dir = os.path.join(train_path, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_dir, "best.pt")

    # --- Merge new images ---
    merged = safe_merge_new_images(images_dir, labels_dir, val_ratio)
    if merged == 0:
        train_dir = os.path.join(images_dir, "train")
        val_dir = os.path.join(images_dir, "val")
        if not os.listdir(train_dir) and not os.listdir(val_dir):
            raise RuntimeError("❌ No training data found! Both train/val are empty.")

    data_yaml_path = update_data_yaml(dataset_dir)

    # --- Device info ---
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Training on device: {device}")

    # --- Train YOLO ---
    try:
        # Determine which model to use
        if os.path.exists(best_weights_path):
            # Continue training using existing best.pt
            model = YOLO(best_weights_path)
            model.to(device)
        else:
            # Only now load base YOLOv8n
            from app.utils import YOLO_WEIGHTS
            model = YOLO(YOLO_WEIGHTS)
            model.to(device)

        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            project=save_dir,
            name=train_name,
            exist_ok=True,
        )

        # Ensure best.pt is always saved
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

def _train():
    """Manual training endpoint logic"""
    print("🚀 Starting YOLO training...")
    
    trainResult = train_yolo_autosplit(
        dataset_dir=DATASET_DIR,
        epochs=100,
        imgsz=640,
        val_ratio=0.2
    )

    print(trainResult)

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