import cv2
import numpy as np
import easyocr
from typing import List, Dict, Tuple, Optional

class PlateRecognizer:
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True):
        """
        Initialize the license plate recognizer.
        
        Args:
            languages: List of languages for OCR
            gpu: Whether to use GPU for inference
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocess the license plate image for better OCR results.
        
        Args:
            plate_img: Cropped license plate image (BGR format)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply some morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return opening
    
    def recognize_plate(self, plate_img: np.ndarray) -> str:
        """
        Recognize text from a license plate image.
        
        Args:
            plate_img: Cropped license plate image (BGR format)
            
        Returns:
            Recognized text as a string
        """
        # Preprocess the plate image
        processed = self.preprocess_plate(plate_img)
        
        # Use EasyOCR to recognize text
        results = self.reader.readtext(processed, detail=0, paragraph=True)
        
        # Join all detected text with spaces
        plate_text = ' '.join(results).strip()
        
        # Post-process the result to make it more license-plate-like
        if plate_text:
            # Remove spaces and special characters, convert to uppercase
            plate_text = ''.join(c for c in plate_text.upper() if c.isalnum())
        
        return plate_text if plate_text else "UNKNOWN"
    
    def recognize_plates_in_frame(self, frame: np.ndarray, plates: List[Dict]) -> List[Dict]:
        """
        Recognize text from multiple license plate detections in a frame.
        
        Args:
            frame: Full frame image (BGR format)
            plates: List of plate detections with 'bbox' keys
            
        Returns:
            List of plates with added 'plate_text' and 'plate_confidence' keys
        """
        results = []
        
        for plate in plates:
            x1, y1, x2, y2 = plate['bbox']
            
            # Extract plate region with boundary checking
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Skip if plate region is too small
            if x2 <= x1 or y2 <= y1:
                plate['plate_text'] = "INVALID_REGION"
                plate['plate_confidence'] = 0.0
                results.append(plate)
                continue
                
            plate_roi = frame[y1:y2, x1:x2]
            
            # Skip if ROI is empty
            if plate_roi.size == 0:
                plate['plate_text'] = "NO_PLATE"
                plate['plate_confidence'] = 0.0
                results.append(plate)
                continue
                
            # Recognize plate text
            plate_text = self.recognize_plate(plate_roi)
            
            # Add results to plate info
            plate['plate_text'] = plate_text
            plate['plate_confidence'] = plate.get('confidence', 0.0)
            
            results.append(plate)
            
        return results
