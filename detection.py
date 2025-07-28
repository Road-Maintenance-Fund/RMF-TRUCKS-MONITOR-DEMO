import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any, Optional
import os

class Detector:
    def __init__(self, model_path: str, plate_model_path: str, conf_threshold: float = 0.3):
        """
        Initialize the YOLO detectors.
        
        Args:
            model_path: Path to the main YOLO model (for trucks)
            plate_model_path: Path to the license plate YOLO model
            conf_threshold: Confidence threshold for detections
        """
        print(f"[DEBUG] Loading main YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        print(f"[DEBUG] Loading plate detection model from: {plate_model_path}")
        self.plate_model = YOLO(plate_model_path)
        self.conf_threshold = conf_threshold
        
        # For main model (trucks)
        self.truck_class_ids = [2, 5, 7]  # COCO: car, bus, truck
        self.class_names = self.model.names
        
        # For plate model
        self.plate_class_id = 0  # Assuming plates.pt has only one class (plate)
        
        print("[DEBUG] Main model classes:", self.class_names)
        print("[DEBUG] Plate model classes:", self.plate_model.names)
        for idx, name in self.class_names.items():
            if 'plate' in name.lower() or 'license' in name.lower() or 'number' in name.lower():
                self.plate_class_id = idx
                print(f"[DEBUG] Found license plate class ID: {idx} ({name})")
                break
                
        if self.plate_class_id == -1:
            print("[WARNING] No license plate class found in the model. Plate detection will be limited.")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect trucks and license plates in the given frame.
        
        Args:
            frame: Input BGR image
            
        Returns:
            Tuple of (trucks, plates) where each is a list of detections with keys:
            - 'bbox': [x1, y1, x2, y2] coordinates
            - 'confidence': Detection confidence
            - 'class_id': Class ID
            - 'class_name': Class name
        """
        # Run YOLO inference for trucks/vehicles
        truck_results = self.model(frame, verbose=False)[0]
        
        trucks = []
        plates = []
        
        if not hasattr(truck_results, 'boxes') or truck_results.boxes is None:
            print("[DEBUG] No vehicle detections in the frame")
            return trucks, plates
        
        # Process each detected vehicle
        for box in truck_results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
                
            class_id = int(box.cls[0])
            if class_id not in self.truck_class_ids:
                continue
                
            # Get truck bbox
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Add truck to results
            truck = {
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class_id': class_id,
                'class_name': self.class_names.get(class_id, 'vehicle')
            }
            trucks.append(truck)
            
            # Extract truck ROI (with some padding)
            h, w = frame.shape[:2]
            pad = 20  # Increased padding to capture more context
            roi_x1 = max(0, x1 - pad)
            roi_y1 = max(0, y1 - pad)
            roi_x2 = min(w, x2 + pad)
            roi_y2 = min(h, y2 + pad)
            
            truck_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if truck_roi.size == 0:
                continue
            
            # Save debug image of the truck ROI
            debug_dir = 'debug_plates'
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/truck_roi_{len(trucks)}.jpg", truck_roi)
                
            # Detect plates in the truck ROI with enhanced settings
            try:
                plate_results = self.plate_model(
                    truck_roi,
                    conf=PLATE_CONFIDENCE,
                    imgsz=640,
                    verbose=False
                )[0]
                
                for plate_box in plate_results.boxes:
                    plate_conf = float(plate_box.conf[0])
                    if plate_conf < PLATE_CONFIDENCE:
                        continue
                        
                    # Convert plate coordinates from ROI to full frame
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0].tolist())
                    
                    # Calculate absolute coordinates
                    abs_x1 = roi_x1 + px1
                    abs_y1 = roi_y1 + py1
                    abs_x2 = roi_x1 + px2
                    abs_y2 = roi_y1 + py2
                    
                    # Calculate dimensions and aspect ratio
                    width = abs_x2 - abs_x1
                    height = abs_y2 - abs_y1
                    area = width * height
                    aspect_ratio = width / max(height, 1e-6)
                    
                    # Skip if plate is too small or has invalid dimensions
                    if width < PLATE_MIN_SIZE or height < PLATE_MIN_SIZE:
                        print(f"[PLATE] Skipping small plate: {width}x{height} (min: {PLATE_MIN_SIZE})")
                        continue
                    
                    # Check aspect ratio
                    if not (PLATE_ASPECT_RATIO[0] <= aspect_ratio <= PLATE_ASPECT_RATIO[1]):
                        print(f"[PLATE] Skipping due to aspect ratio: {aspect_ratio:.2f} (allowed: {PLATE_ASPECT_RATIO})")
                        continue
                    
                    # Log the detection
                    print(f"[PLATE] Detected: {abs_x1}x{abs_y1}-{abs_x2}x{abs_y2} "
                          f"(conf: {plate_conf:.2f}, size: {width}x{height}, area: {area}, ar: {aspect_ratio:.2f})")
                    
                    # Add plate to results
                    plate = {
                        'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                        'confidence': plate_conf,
                        'class_id': self.plate_class_id,
                        'class_name': 'license_plate',
                        'truck_bbox': [x1, y1, x2, y2],  # Reference to parent truck
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    }
                    plates.append(plate)
                    
                    # Save debug image of the detected plate
                    plate_img = truck_roi[py1:py2, px1:px2]
                    if plate_img.size > 0:
                        cv2.imwrite(f"{debug_dir}/plate_{len(plates)}_conf_{plate_conf:.2f}.jpg", plate_img)
                        
            except Exception as e:
                print(f"[ERROR] Plate detection failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Fallback: If no plates found with the plate model, try other methods
        if not plates and trucks:
            print("[DEBUG] No plates found with YOLO, trying fallback detection...")
            fallback_plates = self._fallback_plate_detection(frame, trucks)
            print(f"[DEBUG] Found {len(fallback_plates)} plates with fallback method")
            plates.extend(fallback_plates)
            
        print(f"[DEBUG] Detected {len(plates)} plates in total")
        return trucks, plates
    
    def _fallback_plate_detection(self, frame: np.ndarray, trucks: List[Dict]) -> List[Dict]:
        """Enhanced contour-based license plate localization inside each truck ROI."""
        candidates = []
        h, w = frame.shape[:2]
        debug_dir = 'debug_plates'
        os.makedirs(debug_dir, exist_ok=True)
        
        for i, truck in enumerate(trucks):
            x1, y1, x2, y2 = truck['bbox']
            truck_w, truck_h = x2 - x1, y2 - y1
            
            # Skip very small trucks
            if truck_w < 50 or truck_h < 50:
                continue
                
            # Extract ROI with some padding
            pad_x = int(truck_w * 0.1)
            pad_y = int(truck_h * 0.1)
            roi_x1 = max(0, x1 - pad_x)
            roi_y1 = max(0, y1 - pad_y)
            roi_x2 = min(w, x2 + pad_x)
            roi_y2 = min(h, y2 + pad_y)
            
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0:
                continue
                
            # Save debug image of the truck ROI
            cv2.imwrite(f"{debug_dir}/fallback_truck_{i}.jpg", roi)
            
            # Convert to grayscale and enhance contrast
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for j, cnt in enumerate(contours):
                x, y, cw, ch = cv2.boundingRect(cnt)
                
                # Skip very small or very large contours
                if cw * ch < 100 or cw * ch > (truck_w * truck_h) * 0.5:
                    continue
                    
                aspect_ratio = cw / float(ch + 1e-6)
                
                # Check for typical license plate aspect ratio (wide rectangle)
                if not (2.0 < aspect_ratio < 8.0):
                    continue
                
                # Calculate solidity (area / convex hull area)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                solidity = float(cw * ch) / hull_area
                
                # Filter by solidity (plates are usually solid rectangles)
                if solidity < 0.7:
                    continue
                
                # Map back to full image coordinates
                px1, py1, px2, py2 = x1 + x, y1 + y, x1 + x + cw, y1 + y + ch
                
                # Calculate confidence based on aspect ratio and solidity
                confidence = min(0.8, 0.3 + (min(aspect_ratio, 5.0) / 10.0) + (solidity * 0.3))
                
                candidates.append({
                    'bbox': [px1, py1, px2, py2],
                    'confidence': confidence,
                    'class_id': 999,
                    'class_name': 'license_plate',
                    'truck_bbox': [x1, y1, x2, y2],
                    'method': 'fallback_contour'
                })
                
                # Save debug image of the potential plate
                plate_img = roi[y:y+ch, x:x+cw]
                if plate_img.size > 0:
                    cv2.imwrite(f"{debug_dir}/fallback_plate_{i}_{j}_conf_{confidence:.2f}.jpg", plate_img)
        
        print(f"[DEBUG] Fallback detection found {len(candidates)} potential plates")
        return candidates

    def draw_detections(self, frame: np.ndarray, detections: List[Dict], label: str = None) -> np.ndarray:
        """
        Draw detection bounding boxes and labels on the frame.
        
        Args:
            frame: Input BGR image
            detections: List of detection dictionaries
            label: Optional label to prefix to the class name
            
        Returns:
            Frame with drawn detections
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create label
            text = f"{f'{label} ' if label else ''}{class_name} {conf:.2f}"
            
            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Draw filled rectangle for text background
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_w, y1), (0, 255, 0), -1)
            
            # Put text
            cv2.putText(frame, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return frame
