# This script merges Roboflow datasets into a final format for training.
import shutil
from utils.config import ROBOFLOW, IMAGES_DIR, LABELS_DIR
# Paths

RF_SETS = ["train", "valid", "test"]

# Final dataset folders
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Roboflow → Final class mapping
rf_to_final = {
    0: 3,   # cocci -> coccidiosis poops
    1: 0,   # healthy -> healthy
    2: 7    # salmo -> salmo
}

def convert_label(src, dst):
    new_lines = []

    with open(src, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            cls_id = int(parts[0])
            if cls_id not in rf_to_final:
                continue
            
            parts[0] = str(rf_to_final[cls_id])
            new_lines.append(" ".join(parts))

    if not new_lines:
        return False

    with open(dst, "w") as f:
        f.write("\n".join(new_lines))

    return True

def merge_set(set_name):
    img_dir = ROBOFLOW / set_name / "images"
    label_dir = ROBOFLOW / set_name / "labels"

    for label_file in label_dir.glob("*.txt"):
        stem = label_file.stem
        
        # Find corresponding image
        for ext in ["jpg", "png", "jpeg"]:
            img_path = img_dir / f"{stem}.{ext}"
            if img_path.exists():
                break
        else:
            print(f"⚠ No image for {stem}, skipping")
            continue

        # Output paths
        dst_label = LABELS_DIR / label_file.name
        dst_image = IMAGES_DIR / img_path.name

        # Convert labels
        if convert_label(label_file, dst_label):
            shutil.copy(img_path, dst_image)
            print(f"✔ Merged {set_name}: {img_path.name}")
        else:
            print(f"✘ Skipped {img_path.name} (empty after mapping)")

# Merge all sets
for s in RF_SETS:
    merge_set(s)

print("\n🎉 Merge completed!")
