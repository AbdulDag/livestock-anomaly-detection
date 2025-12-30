import os
from ultralytics import YOLO


script_dir = os.path.dirname(os.path.abspath(__file__))


project_root = os.path.abspath(os.path.join(script_dir, ".."))

model_path = os.path.join(project_root, "runs", "pose", "cow_pose_model8", "weights", "best.pt")
video_path = os.path.join(project_root, "media", "cow_walk2.avi") 


print(f"Loading model from: {model_path}")
print(f"Loading video from: {video_path}")

model = YOLO(model_path)
results = model.track(
    source=video_path,
    save=True,
    show=True,
    conf=0.25,
    tracker="bytetrack.yaml"
)