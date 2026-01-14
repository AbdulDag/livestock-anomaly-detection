from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    #windows freezes for some reason this helps
    multiprocessing.freeze_support() 
    
    
    model = YOLO('yolov8n-pose.pt') 

    # Train
    results = model.train(
        data='data.yaml',
        epochs=100,           
        imgsz=640, #i choose 640 cuz my dataset is small and cows are big in frame
        batch=16, #my gpu can handle 16 cuz of vram dont raise unless you have more vram
        
       
        #augmentations to prevent overfitting, https://docs.ultralytics.com/modes/train/#augmentation-parameters. pretty good for making model robust
        mosaic=1.0,           #use mosaic augmentation which combines 4 images into one
        mixup=0.15,           
        degrees=15.0,         
        scale=0.5,            
        fliplr=0.5,           
        
        #Color alterations
        hsv_h=0.015,           # hue
        hsv_s=0.7,            # saturation
        hsv_v=0.4,            # value for brightness
        
        # Optimization
        patience=20,          #stops training if no improvement after 20 epochs
        device=0, #gpu
        workers=4  # amount of cpu cores to use. i have 12 core cpu but im watching youtube while training so i limit to 4
    )