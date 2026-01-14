import cv2
import os
import numpy as np
from ultralytics import YOLO


CONFIDENCE_THRESHOLD = 0.6  


project_root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_root, "runs", "pose", "train6", "weights", "best.pt")
video_path = os.path.join(project_root, "media", "cow_walk3.mp4")
output_folder = os.path.join(project_root, "media")  

#tracker config file path. the tracker basically is responsible for keeping the IDs consistent across frames
tracker_path = os.path.join(project_root, "custom_tracker.yaml")

#these settings can be found in ht
def create_tracker_file():
    with open(tracker_path, 'w') as f:
        #was using bytetrack before but botsort works better for animal occlusion cuz of its techniques
        f.write("tracker_type: botsort\n") #deeplabcut experience, learned that botsort is good for animals that may hide and reappear.
        f.write("track_high_thresh: 0.5\n") # confidence threshold to start a new track, it doesnt start tracking low confidence detections
        f.write("track_low_thresh: 0.1\n") 
        f.write("new_track_thresh: 0.6\n")
        f.write("track_buffer: 60\n") #forgot what this does exactly
        f.write("match_thresh: 0.8\n")
        f.write("gmc_method: sparseOptFlow\n")
        f.write("proximity_thresh: 0.5\n")
        f.write("appearance_thresh: 0.25\n")
        f.write("with_reid: False\n")
        f.write("fuse_score: True\n") 


if __name__ == "__main__":
    create_tracker_file()
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        exit()
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at {video_path}")
        exit()

    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)

    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    
    save_name = f"labeled_{os.path.basename(video_path)}"
    save_path = os.path.join(output_folder, save_name)
    
    #opencv course showed me this way of saving video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    print(f"Processing... Output will be saved to: {save_path}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break


        results = model.track(frame, persist=True, tracker=tracker_path, verbose=False, imgsz=640)

        if results[0].boxes.id is not None:
            # draws bounding boxes and track IDs, converts gpu tensors to cpu numpy arrays
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
           
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy()
            
            #draw bounding box based on tracking
            for box, track_id, kpts, conf in zip(boxes, track_ids, keypoints, confs):
                x1, y1, x2, y2 = map(int, box)
                
        
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                
                for i, (x, y) in enumerate(kpts):
                    score = conf[i]
                    if score > CONFIDENCE_THRESHOLD:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1) 

    
        out.write(frame)

        cv2.imshow("Filtered Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

   
    cap.release()
    out.release() 
    cv2.destroyAllWindows()
    print("Video saved.")