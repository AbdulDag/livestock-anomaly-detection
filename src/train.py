from ultralytics import YOLO
import os

if __name__ == '__main__':
    
    #Get the folder where THIS script (src/train.py) lives
    script_dir = os.path.dirname(os.path.abspath(__file__))


    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    yaml_path = os.path.join(project_root, "data.yaml")
    base_model_path = os.path.join(project_root, "models", "yolov8n-pose.pt")

    print(f"-------- CONFIGURATION --------")
    print(f"Root Folder:   {project_root}")
    print(f"Data Config:   {yaml_path}")
    print(f"Base Model:    {base_model_path}")
    print(f"-------------------------------")

    #Load the model
    model = YOLO(base_model_path) 

    #Train the model
    results = model.train(
        data=yaml_path,
        epochs=50, 
        imgsz=640, 
        batch=8, 
        name='cow_pose_model'
    )