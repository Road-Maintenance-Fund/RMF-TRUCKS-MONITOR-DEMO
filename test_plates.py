import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os

def test_model():
    # Load the best model
    model_path = "runs/detect/plates_yolov8n/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "runs/detect/plates_yolov8n/weights/last.pt"
    
    model = YOLO(model_path)
    
    # Test on sample images
    test_dir = Path("dataset/test")
    if not test_dir.exists():
        test_dir = Path("dataset/val")
    
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not image_files:
        print("No test images found. Using a sample image...")
        # Create a sample test
        sample_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(sample_img, "Sample Test Image", (50, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        image_files = ["sample.jpg"]
        cv2.imwrite("sample.jpg", sample_img)
    
    # Run inference
    for img_path in image_files[:5]:  # Test on first 5 images
        print(f"\nTesting on {img_path}")
        results = model(str(img_path))
        
        # Show results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            
            # Display in Colab
            from IPython.display import Image, display
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            plt.imshow(im)
            plt.axis('off')
            plt.show()
            
            # Print detections
            for box in r.boxes:
                print(f"Detected: {r.names[box.cls[0].item()]} "
                      f"with confidence {box.conf[0].item():.2f}")

if __name__ == "__main__":
    test_model()
