import os
from ultralytics import YOLO
import argparse

def resume_training(weights_path, data_yaml, epochs, imgsz=640, batch=16, device='0'):
    """
    Resume YOLOv8 training from the last checkpoint.
    
    Args:
        weights_path (str): Path to the last.pt or best.pt file
        data_yaml (str): Path to the data.yaml file
        epochs (int): Total number of epochs to train for (including already completed ones)
        imgsz (int): Image size for training
        batch (int): Batch size
        device (str): Device to use for training (e.g., '0' for GPU 0, 'cpu' for CPU)
    """
    print(f"[INFO] Loading model from {weights_path}")
    
    # Load the model with the last checkpoint
    model = YOLO(weights_path)
    
    # Get the number of completed epochs from the model if available
    completed_epochs = getattr(model, 'epoch', 0)
    print(f"[INFO] Resuming from epoch {completed_epochs}")
    
    # Calculate remaining epochs
    remaining_epochs = max(0, epochs - completed_epochs)
    
    if remaining_epochs <= 0:
        print("[INFO] Training already completed for the specified number of epochs.")
        return
    
    print(f"[INFO] Training for {remaining_epochs} more epochs...")
    
    # Resume training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        resume=True,  # This is the key parameter to resume training
        project='plates_model',
        name='resumed_training',
        exist_ok=True
    )
    
    print("[INFO] Training completed!")
    
    # Export to ONNX
    print("[INFO] Exporting model to ONNX...")
    success = model.export(format='onnx')
    if success:
        print("[INFO] Model exported to ONNX format")
    else:
        print("[WARNING] Failed to export model to ONNX")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resume YOLOv8 training')
    parser.add_argument('--weights', type=str, required=True, 
                        help='Path to the last.pt or best.pt file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Total number of epochs to train for (including completed ones)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use for training (e.g., 0 for GPU 0, cpu for CPU)')
    
    args = parser.parse_args()
    
    resume_training(
        weights_path=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
