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
            print("\n🚀 Starting auto-training process...")
            print("=" * 50)
            
            # Import dependencies with error handling
            try:
                from app.utils import ModelManager, DEVICE, DATASET_DIR, BASE_DIR
            except ImportError as e:
                print(f"❌ Failed to import required modules: {str(e)}")
                raise

            # 1️⃣ Organize data into training folders
            try:
                # Create train directories if they don't exist
                train_img_dir = os.path.join(IMAGES_DIR, "train")
                train_lbl_dir = os.path.join(LABELS_DIR, "train")
                val_img_dir = os.path.join(IMAGES_DIR, "val")
                val_lbl_dir = os.path.join(LABELS_DIR, "val")
                
                for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
                    os.makedirs(d, exist_ok=True)

                # Move images and labels from root folders to train
                img_files = [f for f in os.listdir(IMAGES_DIR) 
                           if os.path.isfile(os.path.join(IMAGES_DIR, f)) and 
                           f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if img_files:
                    print(f"📦 Found {len(img_files)} images to process")
                    
                    for img_file in img_files:
                        # Move image
                        src_img = os.path.join(IMAGES_DIR, img_file)
                        dst_img = os.path.join(train_img_dir, img_file)
                        if os.path.exists(src_img):
                            shutil.move(src_img, dst_img)
                            print(f"✅ Moved image: {img_file}")

                        # Move corresponding label
                        label_file = os.path.splitext(img_file)[0] + ".txt"
                        src_lbl = os.path.join(LABELS_DIR, label_file)
                        dst_lbl = os.path.join(train_lbl_dir, label_file)
                        if os.path.exists(src_lbl):
                            shutil.move(src_lbl, dst_lbl)
                            print(f"✅ Moved label: {label_file}")
                        
                print("✅ Successfully organized data for training")
            except Exception as e:
                print(f"❌ Error merging data: {str(e)}")
                raise

            # 2️⃣ Prepare and validate dataset
            images_dir = os.path.join(DATASET_DIR, "images")
            train_dir = os.path.join(images_dir, "train")
            val_dir = os.path.join(images_dir, "val")
            classes_path = os.path.join(DATASET_DIR, "classes.txt")
            data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")

            # Validate directories exist
            for d in [train_dir, val_dir]:
                if not os.path.exists(d):
                    print(f"⚠️ Directory not found, creating: {d}")
                    os.makedirs(d, exist_ok=True)
            
            # Count training images
            train_images = [f for f in os.listdir(train_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # If no images in train dir, check parent dir
            if not train_images:
                parent_images = [f for f in os.listdir(images_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if parent_images:
                    print(f"🔄 Moving {len(parent_images)} images from parent dir to train dir")
                    for img in parent_images:
                        src = os.path.join(images_dir, img)
                        dst = os.path.join(train_dir, img)
                        if os.path.exists(src):
                            shutil.move(src, dst)
                    train_images = parent_images
            
            if not train_images:
                raise ValueError("No training images found in dataset. Please add images first.")
            
            print(f"📊 Found {len(train_images)} training images")

            # Load and validate classes
            if not os.path.exists(classes_path):
                raise FileNotFoundError(f"Classes file not found: {classes_path}")
                
            with open(classes_path, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]

            nc = len(class_names)
            if nc == 0:
                raise ValueError("No classes found in classes.txt")

            # Create simple data.yaml with only required fields
            yaml_content = {
                "path": DATASET_DIR,  # Dataset root dir
                "train": os.path.join(images_dir, "train"),  # Train images
                "val": os.path.join(images_dir, "val"),      # Val images
                "test": os.path.join(images_dir, "val"),     # Test images (using val set)
                "nc": nc,  # Number of classes
                "names": class_names  # Class names
            }
            
            print("📝 Writing data.yaml with dataset configuration")
            import yaml
            with open(data_yaml_path, "w") as f:
                yaml.safe_dump(yaml_content, f, sort_keys=False, default_flow_style=False)

            # 3️⃣ Get latest trained weights to continue training
            from app.utils import get_latest_trained_weights
            latest_weights = get_latest_trained_weights()
            print(f"🔄 Continuing training from weights: {latest_weights}")
            
            # Create new model instance with latest weights
            model = YOLO(latest_weights)
            print(f"🖥️ Training on {DEVICE.upper()}")

            # 4️⃣ Configure and validate training settings
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"train_{timestamp}"
            
            # Ensure runs directory exists
            runs_dir = os.path.join(BASE_DIR, "runs", "detect")
            os.makedirs(runs_dir, exist_ok=True)
            
            # Validate weights file
            if not os.path.exists(latest_weights):
                raise FileNotFoundError(f"Weights file not found: {latest_weights}")
            
            # Optimize training parameters based on system
            batch_size = 8 if DEVICE == "cuda" else 1
            num_workers = 4 if DEVICE == "cuda" else 0
            
            print(f"💻 Training configuration:")
            print(f"   - Device: {DEVICE}")
            print(f"   - Batch size: {batch_size}")
            print(f"   - Workers: {num_workers}")
            print(f"   - Epochs: {epochs}")
            print(f"   - Image size: {imgsz}")
            print(f"   - Classes: {nc}")
            
            # Use the correct model loading approach
            model = YOLO(latest_weights)
            
            train_args = {
                "data": data_yaml_path,
                "epochs": epochs,
                "imgsz": imgsz,
                "project": runs_dir,
                "name": run_name,
                "exist_ok": True,
                "device": DEVICE,
                "batch": batch_size,
                "workers": num_workers,
                "verbose": True,
                "patience": 50,  # Early stopping patience
                "save_period": 10,  # Save every 10 epochs
                "lr0": 0.001,  # Initial learning rate
                "lrf": 0.01,  # Final learning rate
                "warmup_epochs": 3.0,  # Warmup epochs
                "optimizer": "AdamW",  # Optimizer
                "weight_decay": 0.0005,  # Weight decay
                "momentum": 0.937,  # SGD momentum/Adam beta1
                "cos_lr": True,  # Use cosine LR scheduler
                "close_mosaic": 10,  # Disable mosaic augmentation for final epochs
                "amp": False  # Disable mixed precision training on CPU
            }

            print(f"\n📊 Starting training with configuration:")
            for k, v in train_args.items():
                print(f"   {k}: {v}")
            
            try:
                print("\n🏃 Training in progress...")
                print("=" * 50)
                
                # Start training with the loaded model
                results = model.train(**train_args)
                
                # After training, check for the new best weights
                best_path = os.path.join(runs_dir, run_name, "weights", "best.pt")
                if os.path.exists(best_path):
                    from app import utils
                    print(f"✅ Training completed successfully!")
                    print(f"📊 Results saved to: {os.path.join(runs_dir, run_name)}")
                    
                    # Validate and update weights
                    try:
                        print("🔄 Validating new weights...")
                        test_model = YOLO(best_path)
                        
                        # Update the model
                        with reload_lock:
                            utils.TRAINED_WEIGHTS = best_path
                            utils.yolo = test_model
                            print(f"✅ Model updated with new weights: {best_path}")
                    except Exception as e:
                        print(f"⚠️ Warning: New weights validation failed: {str(e)}")
                        print("⚠️ Continuing with previous weights")
                else:
                    print("⚠️ Warning: No best.pt found after training")
                    print("⚠️ Continuing with previous weights")
                    
            except Exception as e:
                print(f"❌ Training failed: {str(e)}")
                print("⚠️ Model will continue using previous weights")
                import traceback
                traceback.print_exc()
                raise

        except Exception as e:
            import traceback
            print("❌ Auto-train failed:")
            traceback.print_exc()

    threading.Thread(target=_run, daemon=True).start()