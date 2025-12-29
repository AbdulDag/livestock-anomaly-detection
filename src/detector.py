#test if its detecting cows from ultralytics YOLOv8 model

import cv2
from ultralytics import YOLO

class CowDetector:
    def __init__(self, model_path=None):
        
        if model_path:
            self.model = YOLO(model_path)
            print(f"Loaded custom model from: {model_path}")
        else:
            # Load standard pre-trained model just for testing
            self.model = YOLO("yolov8n.pt") 
            print("Loaded base YOLOv8 model for testing.")

    def detect(self, frame, conf_threshold=0.5):
      
        #19 for cow class
        results = self.model(frame, classes=[19], conf=conf_threshold, verbose=False)    
        detections = []

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                
                # Append to our list
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': confidence
                })

        return detections


if __name__ == "__main__":
    detector = CowDetector() 
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    print("Starting Test Loop... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cow_boxes = detector.detect(frame)

        for cow in cow_boxes:
            x1, y1, x2, y2 = cow['box']
            conf = cow['conf']
            
            # Draw rectangle: Green color (0, 255, 0), thickness 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label
            label = f"Cow: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Detector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()