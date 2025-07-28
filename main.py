import os
import cv2
import time
import json
import numpy as np
import streamlit as st
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Import our modules
from detection import Detector
from ocr import PlateRecognizer
from classify import TruckClassifier
from tracking import Tracker
from config import *

class TruckMonitoringSystem:
    def __init__(self):
        """Initialize the truck monitoring system."""
        # Initialize models
        self.detector = Detector(
            model_path=os.path.join(MODEL_DIR, YOLO_MODEL),
            plate_model_path=os.path.join(MODEL_DIR, PLATE_MODEL),
            conf_threshold=CONFIDENCE_THRESHOLD
        )
        self.ocr = PlateRecognizer(
            languages=OCR_LANGUAGES,
            gpu=OCR_GPU
        )
        # Initialize classifier with the same model as detector for consistency
        self.classifier = TruckClassifier(model_path=os.path.join(MODEL_DIR, YOLO_MODEL))
        # Initialize tracker
        self.tracker = Tracker(os.path.join(MODEL_DIR, YOLO_MODEL))
        # Keep track of plate IDs already read
        self.seen_plate_ids = set()
        
        # Detection history
        self.detection_history = []
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame to detect trucks and license plates."""
        try:
            # Make a copy of the frame for drawing
            annotated_frame = frame.copy()
            
            # Detect trucks and plates
            trucks, plates = self.detector.detect(frame)
            
            # Track objects - using the tracker's track method which returns (annotated_frame, detections)
            try:
                annotated_frame, tracked_objects = self.tracker.track(frame)
            except Exception as e:
                print(f"[ERROR] Tracking failed: {e}")
                tracked_objects = []
            
            # Prepare results
            results = []
            
            # Process each detected truck
            for obj in tracked_objects:
                try:
                    # Get truck data
                    truck = {
                        'id': int(obj.get('id', 0)),
                        'bbox': [int(coord) for coord in obj.get('bbox', [0, 0, 0, 0])],
                        'class_name': 'truck',
                        'classification': 'single',  # Default classification
                        'classification_confidence': 0.0,
                        'plates': []
                    }
                    
                    # Find associated plates
                    x1, y1, x2, y2 = truck['bbox']
                    truck_roi = frame[max(0, y1):min(frame.shape[0], y2), 
                                    max(0, x1):min(frame.shape[1], x2)]
                    
                    # Find plates that belong to this truck
                    for plate in plates:
                        px1, py1, px2, py2 = map(int, plate.get('bbox', [0, 0, 0, 0]))
                        plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                        
                        # Check if plate center is inside truck bbox
                        if (x1 <= plate_center[0] <= x2 and y1 <= plate_center[1] <= y2):
                            # Run OCR if we haven't seen this plate before
                            plate_id = plate.get('id', id(tuple(plate.get('bbox', []))))
                            if plate_id not in self.seen_plate_ids:
                                try:
                                    plate_img = frame[max(0, py1):min(frame.shape[0], py2), 
                                                   max(0, px1):min(frame.shape[1], px2)]
                                    if plate_img.size > 0:
                                        try:
                                            plate_text = self.ocr.recognize_plate(plate_img)
                                            # If we get a tuple (text, confidence), unpack it
                                            if isinstance(plate_text, tuple) and len(plate_text) == 2:
                                                plate_text, confidence = plate_text
                                            else:
                                                confidence = 0.8  # Default confidence if not provided
                                            
                                            plate['plate_text'] = plate_text.strip() if plate_text else 'UNKNOWN'
                                            plate['confidence'] = float(confidence)
                                            print(f"[INFO] Detected plate: {plate['plate_text']} with confidence {confidence:.2f}")
                                            self.seen_plate_ids.add(plate_id)
                                        except Exception as e:
                                            print(f"[ERROR] Plate recognition failed: {e}")
                                            plate['plate_text'] = 'OCR_ERROR'
                                            plate['confidence'] = 0.0
                                except Exception as e:
                                    print(f"[ERROR] OCR failed: {e}")
                                    plate['plate_text'] = 'OCR_ERROR'
                                    plate['confidence'] = 0.0
                            
                            truck['plates'].append({
                                'bbox': [int(px1), int(py1), int(px2), int(py2)],
                                'plate_text': plate.get('plate_text', 'UNKNOWN'),
                                'confidence': float(plate.get('confidence', 0.0))
                            })
                    
                    # Classify truck type if we have plates or a good detection
                    if truck['plates'] or (x2 - x1) * (y2 - y1) > 1000:  # Only classify if we have plates or a reasonably sized truck
                        try:
                            truck_img = truck_roi
                            if truck_img.size > 0:
                                classification, confidence = self.classifier.classify(truck_img)
                                truck['classification'] = str(classification).lower()
                                truck['classification_confidence'] = float(confidence)
                        except Exception as e:
                            print(f"[ERROR] Classification failed: {e}")
                            truck['classification'] = 'unknown'
                            truck['classification_confidence'] = 0.0
                    
                    # Add to results if we have plates or good classification confidence
                    if truck['plates'] or truck['classification_confidence'] > 0.3:
                        results.append(truck)
                        
                except Exception as e:
                    print(f"[ERROR] Error processing truck detection: {e}")
                    continue
            
            # Draw results on frame
            if results:
                try:
                    annotated_frame = self._draw_results(annotated_frame, results)
                except Exception as e:
                    print(f"[ERROR] Error drawing results: {e}")
            
            # Log detections
            if results:
                self._log_detections(results)
            
            return annotated_frame, results
            
        except Exception as e:
            print(f"[CRITICAL] Error in process_frame: {e}")
            return frame, []
    
    def _draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Draw detection results on the frame."""
        for truck in results:
            # Draw truck bounding box
            x1, y1, x2, y2 = map(int, truck['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get truck class and confidence
            truck_class = truck.get('class_name', 'truck').upper()
            conf = truck.get('classification_confidence', 0)
            
            # Draw truck class and confidence
            label = f"{truck_class} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw plates
            for plate in truck.get('plates', []):
                px1, py1, px2, py2 = map(int, plate['bbox'])
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                
                # Get plate text and confidence
                plate_text = plate.get('plate_text', 'UNKNOWN')
                conf = plate.get('confidence', 0)
                
                # Draw plate text with background for better visibility
                text = f"{plate_text} ({conf:.2f})" if plate_text != 'UNKNOWN' else 'UNKNOWN'
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (px1, py1 - text_h - 10), (px1 + text_w, py1), (0, 0, 255), -1)
                cv2.putText(frame, text, (px1, py1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _log_detections(self, results: List[Dict]):
        """Log all detections with their details."""
        if not results:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entries = []

        for truck in results:
            # Prepare plate information
            plates_info = []
            for plate in truck.get('plates', []):
                plate_text = plate.get('plate_text', '').strip()
                if plate_text and plate_text != 'UNKNOWN':
                    plates_info.append({
                        'text': plate_text,
                        'confidence': float(plate.get('confidence', 0.0)),
                        'bbox': list(map(int, plate.get('bbox', [])))
                    })

            # Create log entry for this detection
            log_entry = {
                'timestamp': timestamp,
                'truck_id': int(truck.get('id', 0)),
                'class_name': str(truck.get('class_name', 'truck')),
                'classification': str(truck.get('classification', 'unknown')),
                'confidence': float(truck.get('classification_confidence', 0.0)),
                'bbox': list(map(int, truck.get('bbox', []))),
                'plates': plates_info
            }
            
            log_entries.append(log_entry)
            self.detection_history.append(log_entry)
            
            # Log to file
            try:
                with open(LOG_FILE, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                print(f"[LOG] Logged detection: {log_entry}")
            except Exception as e:
                print(f"[ERROR] Failed to write to log file: {e}")
                
        return log_entries

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title=STREAMLIT_TITLE,
        layout=STREAMLIT_LAYOUT
    )
    
    st.title(STREAMLIT_TITLE)
    
    # Initialize session state
    if 'system' not in st.session_state:
        st.session_state.system = TruckMonitoringSystem()
    
    # Sidebar controls
    st.sidebar.title(STREAMLIT_SIDEBAR_TITLE)
    
    # Input selection
    input_type = st.sidebar.radio(
        "Input Source",
        ["Webcam", "Video File", "Image"]
    )
    
    # Initialize variables
    frame_placeholder = st.empty()
    results_placeholder = st.empty()
    
    if input_type == "Webcam":
        # Webcam capture
        cap = cv2.VideoCapture(0)
        
        stop_button = st.sidebar.button("Stop")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
                
            # Process frame
            processed_frame, results = st.session_state.system.process_frame(frame)
            
            # Display results
            frame_placeholder.image(
                cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )
            
            # Show results in sidebar
            display_results(results)
            
            # Check for stop button click
            if stop_button:
                break
            
            # Small delay to prevent high CPU usage
            time.sleep(0.03)
            
        cap.release()
        
    elif input_type == "Video File":
        # Video file upload
        video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        
        if video_file is not None:
            # Save uploaded file temporarily
            temp_file = os.path.join(DATA_DIR, "temp_video.mp4")
            with open(temp_file, "wb") as f:
                f.write(video_file.read())
            
            cap = cv2.VideoCapture(temp_file)
            
            play_button = st.sidebar.button("Play")
            stop_button = st.sidebar.button("Stop")
            
            while play_button and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.warning("End of video")
                    break
                    
                # Process frame
                processed_frame, results = st.session_state.system.process_frame(frame)
                
                # Display results
                frame_placeholder.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )
                
                # Show results in sidebar
                display_results(results)
                
                # Small delay to prevent high CPU usage
                time.sleep(0.03)
                
            cap.release()
            
    elif input_type == "Image":
        # Image upload
        image_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        
        if image_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process image
            processed_frame, results = st.session_state.system.process_frame(frame)
            
            # Display results
            frame_placeholder.image(
                cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )
            
            # Show results in sidebar
            display_results(results)
    
def display_results(results: List[Dict]):
    """Display detection results in the sidebar."""
    st.sidebar.subheader("Detection Results")
    
    if not results:
        st.sidebar.info("No trucks detected.")
        return
        
    for i, truck in enumerate(results, 1):
        with st.sidebar.expander(f"Truck {i}: {truck['classification'].title()}"):
            st.write(f"**Confidence:** {truck.get('classification_confidence', 0):.2f}")
            
            if 'plates' in truck and truck['plates']:
                st.subheader("License Plates:")
                for j, plate in enumerate(truck['plates'], 1):
                    st.write(f"**Plate {j}:** {plate.get('plate_text', 'UNKNOWN')}")
                    st.write(f"  - Confidence: {plate.get('confidence', 0):.2f}")
            else:
                st.info("No license plates detected for this truck.")

if __name__ == "__main__":
    main()
