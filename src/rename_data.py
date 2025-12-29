import os

IMG_DIR = r"C:\Users\dagab\Desktop\yoloCow\cow-pose-estimation-1\train\images"
LABEL_DIR = r"C:\Users\dagab\Desktop\yoloCow\cow-pose-estimation-1\train\labels"


PREFIX = "cow" 
# =================================================

def rename_dataset():
    # Get list of all images (jpg, png, jpeg)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_extensions)]
    
    # Sort them so the order stays deterministic
    images.sort()
    
    count = 0
    errors = 0

    print(f"Found {len(images)} images. Starting rename...")

    for i, filename in enumerate(images):
        # 1. Setup old paths
        old_img_path = os.path.join(IMG_DIR, filename)
        
        name_no_ext, ext = os.path.splitext(filename)
        old_label_name = name_no_ext + ".txt"
        old_label_path = os.path.join(LABEL_DIR, old_label_name)

        # 2. Setup new names (using 4 digits padding: 0001, 0002)
        new_name_base = f"{PREFIX}{i+1:04d}"
        new_img_name = new_name_base + ext
        new_label_name = new_name_base + ".txt"

        new_img_path = os.path.join(IMG_DIR, new_img_name)
        new_label_path = os.path.join(LABEL_DIR, new_label_name)

        # 3. Rename Image
        try:
            os.rename(old_img_path, new_img_path)
        except FileExistsError:
            print(f"Error: {new_img_name} already exists. Skipping.")
            errors += 1
            continue

        # 4. Rename Label (if it exists)
        if os.path.exists(old_label_path):
            try:
                os.rename(old_label_path, new_label_path)
            except FileExistsError:
                print(f"Error: {new_label_name} already exists.")
        else:
            # If no label file exists, just warn
            print(f"Warning: No label found for {filename}")

        count += 1

    print(f"Done! Renamed {count} image/label pairs.")
    if errors > 0:
        print(f"Encountered {errors} errors (checked filenames probably already existed).")

if __name__ == "__main__":
    # Double check inputs before running
    if not os.path.exists(IMG_DIR) or not os.path.exists(LABEL_DIR):
        print("Error: One of your directory paths is wrong.")
    else:
        rename_dataset()