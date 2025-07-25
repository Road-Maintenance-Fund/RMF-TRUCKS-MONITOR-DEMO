# Truck Monitoring System

A Python-based prototype that uses computer vision to detect trucks and recognize license plates from webcam or video input. The system can identify single and double/articulated trucks based on the number of license plates detected.

## Features

- Real-time truck and license plate detection using YOLOv8
- License plate text recognition using EasyOCR
- Classification of trucks as single or double/articulated
- Support for webcam, video file, and image inputs
- Interactive Streamlit web interface
- Detection logging to a local JSON file

## Prerequisites

- Python 3.8 or higher
- Webcam (for live demo)
- NVIDIA GPU (recommended for better performance)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/truck-monitoring-system.git
   cd truck-monitoring-system
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. Start the Streamlit application:
   ```bash
   streamlit run main.py
   ```

2. The application will open in your default web browser.

3. In the sidebar, select your preferred input source:
   - **Webcam**: Use your computer's camera
   - **Video File**: Upload a video file (MP4, AVI, MOV)
   - **Image**: Upload a single image (JPG, JPEG, PNG)

4. View the detection results in the main window and the sidebar.

### Understanding the Output

- **Trucks** are highlighted with green bounding boxes
- **License plates** are highlighted with red bounding boxes
- The classification (single or double/articulated) is displayed above each truck
- Detection results are logged to `data/detections.log`

## Project Structure

```
truck-monitoring-system/
├── config.py           # Configuration settings
├── detection.py        # YOLO-based object detection
├── ocr.py              # License plate text recognition
├── classify.py         # Truck classification logic
├── main.py             # Main application with Streamlit UI
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Customization

You can modify the following parameters in `config.py` to fine-tune the system:

- `CONFIDENCE_THRESHOLD`: Minimum confidence score for detections
- `PLATE_CONFIDENCE`: Minimum confidence for license plate detection
- `OCR_LANGUAGES`: Languages for text recognition
- `OCR_GPU`: Enable/disable GPU acceleration for OCR

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for computer vision operations
