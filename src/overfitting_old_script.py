from ultralytics import YOLO
import os

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    yaml_path = os.path.join(project_root, "data.yaml")
    base_model_path = os.path.join(project_root, "models", "yolov8n-pose.pt")

    #Load the model
    model = YOLO(base_model_path) 

    #in this version it was overfitting. the precision recall curve was 0.995 which is suspiciously high for my small dataset
    ##that;s why i wrote a new version that uses mosaic augmentation and other tricks to make it more robust
    results = model.train(
        data=yaml_path,
        epochs=50, 
        imgsz=640, 
        batch=8, 
        name='cow_pose_model' 
    )