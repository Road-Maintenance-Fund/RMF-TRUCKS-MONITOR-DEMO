import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# YOLO model configuration
# Available models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
# Larger models are more accurate but slower
YOLO_MODEL = 'yolov8s.pt'  # For truck/vehicle detection
PLATE_MODEL = 'plates.pt'   # Specialized model for license plate detection
CONFIDENCE_THRESHOLD = 0.3  # Confidence threshold for vehicle detection

# License plate detection settings
PLATE_CONFIDENCE = 0.5  # Minimum confidence for license plate detection
PLATE_ASPECT_RATIO = (2, 5)  # Expected aspect ratio range for license plates (width/height)
PLATE_MIN_SIZE = 20  # Minimum size (pixels) for plate detection

# OCR settings
OCR_LANGUAGES = ['en']  # Languages for OCR
OCR_GPU = True  # Set to False if running on CPU

# Display settings
SHOW_CONFIDENCE = True
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)  # Green
BOX_COLOR = (0, 255, 0)   # Green
BOX_THICKNESS = 2

# Logging
LOG_FILE = os.path.join(DATA_DIR, 'detections.log')

# Streamlit settings
STREAMLIT_TITLE = "Truck Monitoring System"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_SIDEBAR_TITLE = "Controls"
