import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# YOLO model configuration
# Using the newly trained models from Colab training
YOLO_MODEL = 'yolov8s.pt'  # For truck/vehicle detection
PLATE_MODEL = 'best.pt'     # Using the newly trained plate detection model
CONFIDENCE_THRESHOLD = 0.25  # Lowered threshold for better detection recall

# License plate detection settings
PLATE_CONFIDENCE = 0.25      # Lower confidence threshold for better plate detection recall
PLATE_ASPECT_RATIO = (1.0, 5.0)  # Wider aspect ratio range to catch more plates
PLATE_MIN_SIZE = 20         # Lower minimum size to detect smaller plates

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
