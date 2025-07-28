# Resuming YOLOv8 Training

This guide explains how to resume training your YOLOv8 model from the last checkpoint.

## Prerequisites

- Python 3.8 or later
- PyTorch 1.8.0 or later
- Ultralytics YOLOv8
- Your training data in YOLO format
- The model checkpoint files (`last.pt` and `best.pt`)

## Local Training

1. **Install Required Packages**:
   ```bash
   pip install ultralytics
   ```

2. **Prepare Your Files**:
   - Place your `last.pt` and `best.pt` files in a directory (e.g., `plates_model/weights/`)
   - Make sure your `data.yaml` file is in the correct location

3. **Run the Resume Script**:
   ```bash
   python resume_training.py \
     --weights path/to/plates_model/weights/last.pt \
     --data path/to/data.yaml \
     --epochs 100 \
     --batch 16 \
     --imgsz 640 \
     --device 0  # Use 0 for GPU, 'cpu' for CPU
   ```

## Google Colab Training

1. **Open the Colab Notebook**:
   - Open `resume_training_colab.ipynb` in Google Colab
   - Make sure to select a GPU runtime (Runtime > Change runtime type > GPU)

2. **Upload Your Files**:
   - Run the notebook cells to mount Google Drive
   - When prompted, upload your `last.pt`, `best.pt`, and `data.yaml` files

3. **Start Training**:
   - The notebook will automatically move files to the correct locations
   - Run the training cell to resume training

4. **Save Your Model**:
   - After training, the model will be saved to your Google Drive

## Training Parameters

- `--weights`: Path to the last checkpoint file (usually `last.pt`)
- `--data`: Path to your `data.yaml` file
- `--epochs`: Total number of epochs (including completed ones)
- `--batch`: Batch size (adjust based on your GPU memory)
- `--imgsz`: Input image size (default: 640)
- `--device`: Device to use for training (0 for GPU, 'cpu' for CPU)

## Monitoring Training

- Training progress will be displayed in the console
- TensorBoard logs are saved in the `runs` directory
- The best and last checkpoints are saved automatically

## Troubleshooting

- If you get CUDA out of memory errors, reduce the batch size
- Make sure your `data.yaml` file has the correct paths to your dataset
- Check that your checkpoint files are not corrupted
