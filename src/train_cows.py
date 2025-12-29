from ultralytics import YOLO
import os

#pointing to the data.yaml file
data_path = os.path.abspath("./cow-pose-estimation-1/data.yaml")
model = YOLO('yolov8n-pose.pt') 

#Training the model
results = model.train(
    data=data_path,
    epochs=50, # an epoch is one complete pass through the training dataset
    imgsz=640, 
    batch=8,   # depends on my gpu
    name='cow_pose_model'
)