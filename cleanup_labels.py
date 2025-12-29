import os

IMG_DIR = r"C:\Users\dagab\Desktop\yoloCow\cow-pose-estimation-1\train\images"
LABEL_DIR = r"C:\Users\dagab\Desktop\yoloCow\cow-pose-estimation-1\train\labels"

def cleanup_orphaned_labels():
    #get lst of all labels
    if not os.path.exists(LABEL_DIR):
        print(f"Error: Label directory not found: {LABEL_DIR}")
        return

    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    
    print(f"Scanning {len(label_files)} labels for missing images...")
    
    deleted_count = 0
    kept_count = 0

  
    for label_file in label_files:
        # get filename without extension
        name_no_ext = os.path.splitext(label_file)[0]
        label_path = os.path.join(LABEL_DIR, label_file)
        
        # 3. Look for a matching image in images folder
        image_exists = False
        #check all .parts to just in case
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            img_name = name_no_ext + ext
            img_path = os.path.join(IMG_DIR, img_name)
            if os.path.exists(img_path):
                image_exists = True
                break
        
        #delete if no image found
        if not image_exists:
            try:
                os.remove(label_path)
                print(f"Deleted orphan label: {label_file}")
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {label_file}: {e}")
        else:
            kept_count += 1

    print("-" * 30)
    print("Cleanup Complete.")
    print(f"Kept Labels:    {kept_count}")
    print(f"Deleted Labels: {deleted_count}")

if __name__ == "__main__":
    cleanup_orphaned_labels()