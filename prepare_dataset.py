import os
import yaml
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

def verify_dataset_structure(data_dir):
    """Verify the dataset structure and count annotations."""
    print("Verifying dataset structure...")
    
    # Check if directories exist
    image_dir = Path(data_dir) / 'images'
    label_dir = Path(data_dir) / 'labels'
    
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Dataset directories not found. Expected:\n"
            f"- {image_dir}\n"
            f"- {label_dir}"
        )
    
    # Get list of image and label files
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    label_files = sorted(label_dir.glob('*.txt'))
    
    print(f"Found {len(image_files)} images and {len(label_files)} label files")
    
    # Verify corresponding files exist
    missing_labels = 0
    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"Missing label for {img_path.name}")
            missing_labels += 1
    
    if missing_labels > 0:
        print(f"\nWarning: {missing_labels} images are missing label files")
    
    return image_files, label_files

def split_dataset(image_files, val_split=0.2, test_split=0.1, seed=42):
    """Split dataset into train/val/test sets."""
    # First split: separate test set
    train_val_files, test_files = train_test_split(
        image_files, 
        test_size=test_split,
        random_state=seed
    )
    
    # Second split: separate train and validation
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_split/(1-test_split),  # Adjust for test split
        random_state=seed
    )
    
    print(f"\nDataset split:")
    print(f"- Training: {len(train_files)} images")
    print(f"- Validation: {len(val_files)} images")
    print(f"- Test: {len(test_files)} images")
    
    return train_files, val_files, test_files

def create_yaml(data_dir, train_files, val_files, test_files):
    """Create YAML configuration file for YOLO training."""
    data = {
        'train': str(Path(data_dir) / 'train'),
        'val': str(Path(data_dir) / 'val'),
        'test': str(Path(data_dir) / 'test'),
        'nc': 1,  # Number of classes
        'names': ['license_plate']  # Class names
    }
    
    with open(Path(data_dir) / 'data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"\nCreated dataset configuration at {Path(data_dir)/'data.yaml'}")

def main():
    # Configuration
    data_dir = Path('dataset')
    
    try:
        # Verify dataset
        image_files, _ = verify_dataset_structure(data_dir)
        
        # Split dataset
        train_files, val_files, test_files = split_dataset(image_files)
        
        # Create YAML configuration
        create_yaml(data_dir, train_files, val_files, test_files)
        
        print("\nDataset preparation complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
