import os

print("🔍 Scanning your project for 'train' folders...")
found = False

# Walk through all folders starting from the current directory (yoloCow)
for root, dirs, files in os.walk("."):
    if "train" in dirs:
        full_path = os.path.abspath(os.path.join(root, "train"))
        print(f"\nFOUND TRAIN FOLDER AT:")
        print(f"{full_path}")
        found = True

if not found:
    print("\nCould not find any folder named 'train' inside this directory.")