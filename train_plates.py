import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def train_plate_detector():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the YOLOv8n model (nano version for better speed)
    model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8n model
    
    # Update model to single class (license plates)
    model.overrides['names'] = {0: 'license_plate'}
    
    # Training configuration
    config = {
        'data': 'dataset/data.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 0 if device == 'cuda' else 'cpu',
        'workers': 4,
        'optimizer': 'auto',  # Auto-select best optimizer
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final OneCycleLR learning rate
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # Optimizer weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup initial momentum
        'warmup_bias_lr': 0.1,  # Warmup initial bias lr
        'box': 0.05,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # Distribution Focal Loss gain
        'fl_gamma': 0.0,  # Focal loss gamma
        'hsv_h': 0.015,  # Image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # Image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # Image HSV-Value augmentation (fraction)
        'degrees': 0.0,  # Image rotation (+/- deg)
        'translate': 0.1,  # Image translation (+/- fraction)
        'scale': 0.5,  # Image scale (+/- gain)
        'shear': 0.0,  # Image shear (+/- deg)
        'perspective': 0.0,  # Image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,  # Image flip up-down (probability)
        'fliplr': 0.5,  # Image flip left-right (probability)
        'mosaic': 1.0,  # Image mosaic (probability)
        'mixup': 0.0,  # Image mixup (probability)
        'copy_paste': 0.0,  # Segment copy-paste (probability)
        'name': 'plates_yolov8n',  # Save results to project/name
        'exist_ok': True,  # Existing project/name ok, do not increment
        'pretrained': True,  # Use pretrained weights
        'freeze': 10,  # Freeze first 10 layers
    }
    
    try:
        # Train the model
        results = model.train(**config)
        
        # Export to ONNX for better inference speed
        model.export(format='onnx', dynamic=True)
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {model.export_dir}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_plate_detector()
