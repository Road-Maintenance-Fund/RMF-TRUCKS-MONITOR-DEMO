# License Plate Detection Model Training Guide

This guide explains how to train a lightweight and accurate license plate detection model using YOLOv8.

## Prerequisites

1. Python 3.8 or higher
2. CUDA-enabled GPU (recommended) or CPU
3. Sufficient disk space for the dataset and model checkpoints

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

1. Organize your dataset in the following structure:
   ```
   dataset/
   ├── images/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── labels/
       ├── image1.txt
       ├── image2.txt
       └── ...
   ```

2. Run the dataset preparation script:
   ```bash
   python prepare_dataset.py
   ```

## Training

1. Start the training process:
   ```bash
   python train_plates.py
   ```

2. Monitor training progress:
   - TensorBoard logs are saved in `runs/detect/`
   - Model checkpoints are saved in `runs/detect/plates_yolov8n/`

## Export to ONNX

The training script automatically exports the model to ONNX format after training completes. The ONNX model will be saved in the same directory as the trained weights.

## Integration

To use the trained model in your application:

1. Copy the exported ONNX model to the `models` directory
2. Update the model path in `config.py`
3. The model will be automatically loaded by the `Detector` class

## Tips for Better Performance

1. **Data Quality**: Ensure your training data is clean and well-annotated
2. **Augmentation**: Tweak the augmentation parameters in `dataset/augmentation.yaml`
3. **Hyperparameters**: Adjust learning rate, batch size, and other parameters in `train_plates.py`
4. **Model Size**: For even lighter models, try YOLOv8nano or YOLOv8tiny

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size in `train_plates.py`
- **Slow Training**: Ensure you're using a GPU and the latest CUDA drivers
- **Poor Accuracy**: Check your dataset quality and consider adding more training examples
