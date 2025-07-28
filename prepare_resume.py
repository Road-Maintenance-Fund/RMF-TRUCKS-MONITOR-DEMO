import os
import shutil
from datetime import datetime

def prepare_resume_training(source_dir, target_dir='plates_model'):
    """
    Prepare the directory structure for resuming training.
    
    Args:
        source_dir (str): Directory containing the trained model files
        target_dir (str): Target directory for the training files
    """
    # Create target directory structure
    weights_dir = os.path.join(target_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    # Check for required files
    required_files = {
        'last.pt': os.path.join('runs', 'detect', 'truck_plate_detection', 'weights', 'last.pt'),
        'best.pt': os.path.join('runs', 'detect', 'truck_plate_detection', 'weights', 'best.pt'),
        'data.yaml': 'data.yaml'
    }
    
    # Copy files
    for name, rel_path in required_files.items():
        src = os.path.join(source_dir, rel_path)
        dst = os.path.join(weights_dir if name.endswith('.pt') else target_dir, name)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[INFO] Copied {src} to {dst}")
        else:
            print(f"[WARNING] Could not find {src}")
    
    print("\n[INFO] Preparation complete!")
    print(f"To resume training, run:")
    print(f"python resume_training.py \\")
    print(f"  --weights {os.path.join(weights_dir, 'last.pt')} \\")
    print(f"  --data {os.path.join(target_dir, 'data.yaml')} \\")
    print(f"  --epochs 100  # Adjust as needed")
    print(f"  --batch 16    # Adjust based on your GPU memory")
    print(f"  --device 0    # Use 'cpu' if no GPU is available")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare files for resuming YOLOv8 training')
    parser.add_argument('--source', type=str, default='.',
                        help='Directory containing the trained model files')
    parser.add_argument('--target', type=str, default='plates_model',
                        help='Target directory for the training files')
    
    args = parser.parse_args()
    prepare_resume_training(args.source, args.target)
