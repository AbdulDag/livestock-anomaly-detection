import os
import yaml

base_path = r"C:\Users\dagab\Desktop\yoloCow\cow-pose-estimation-1"
yaml_file = os.path.join(base_path, "data.yaml")

deep_path = os.path.join(base_path, "data", "detection_v1_single")

def find_image_dir(split_name):
    # Check for 'train/images'
    path_with_images = os.path.join(deep_path, split_name, "images")
    if os.path.exists(path_with_images):
        return path_with_images
    
    # Check for just 'train'
    path_direct = os.path.join(deep_path, split_name)
    if os.path.exists(path_direct):
        return path_direct
    
    return None

real_train = find_image_dir("train")
real_val = find_image_dir("valid")

if not real_train or not real_val:
    print("ERROR: Could not find the image folders on your disk.")
    print(f"Checked inside: {deep_path}")
    exit()

print(f"Found correct train path: {real_train}")
print(f"Found correct valid path: {real_val}")

# 3. Read the existing YAML to keep class names
with open(yaml_file, 'r') as f:
    content = yaml.safe_load(f)

content['train'] = real_train
content['val'] = real_val

with open(yaml_file, 'w') as f:
    yaml.dump(content, f)

print(f"\n Successfully updated {yaml_file}")
print("You can now run your training script again.")