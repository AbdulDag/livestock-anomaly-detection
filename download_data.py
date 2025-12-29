from roboflow import Roboflow
#hide this later
rf = Roboflow(api_key="ROBOFLOW_API_KEY")
project = rf.workspace("mikaelapisani").project("cow-pose-estimation-fxosp")
dataset = project.version(1).download("yolov8")