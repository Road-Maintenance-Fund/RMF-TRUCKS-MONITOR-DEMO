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
        Enhanced preprocessing for license plate images to improve OCR accuracy.
        
        Args:
            plate_img: Cropped license plate image (BGR format)
            
        Returns:
            Preprocessed grayscale image optimized for OCR
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Apply dilation to make characters more solid
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            processed = cv2.dilate(opening, kernel, iterations=1)
            
            # Optional: Uncomment to save preprocessed images for debugging
            # cv2.imwrite(f'debug_preprocess_{int(time.time())}.jpg', processed)
            
            return processed
            
        except Exception as e:
            print(f"[OCR] Preprocessing error: {str(e)}")
            # Return the grayscale image if preprocessing fails
            return cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
    
    def recognize_plate(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text from a license plate image with confidence score.
        
        Args:
            plate_img: Cropped license plate image (BGR format)
            
        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        try:
            # Skip if image is too small
            if plate_img.size < 100:  # Less than 10x10 pixels
                return "TOO_SMALL", 0.0
                
            # Preprocess the plate image
            processed = self.preprocess_plate(plate_img)
            
            # Use EasyOCR to recognize text with confidence scores
            results = self.reader.readtext(processed, detail=1, paragraph=False)
            
            if not results:
                return "NO_TEXT", 0.0
                
            # Filter results by confidence and sort by x-coordinate (left to right)
            valid_results = [r for r in results if r[2] > 0.3]  # Minimum confidence
            valid_results.sort(key=lambda x: x[0][0][0])  # Sort by x-coordinate of bbox
            
            # Calculate average confidence
            if valid_results:
                avg_confidence = sum(r[2] for r in valid_results) / len(valid_results)
                plate_text = ' '.join([r[1] for r in valid_results]).strip()
                
                # Post-process the result
                if plate_text:
                    # Remove spaces and special characters, convert to uppercase
                    plate_text = ''.join(c for c in plate_text.upper() if c.isalnum())
                    
                    # Additional validation for plate text
                    if len(plate_text) < 3:  # Too short to be a valid plate
                        return "INVALID_LENGTH", avg_confidence
                        
                    return plate_text, avg_confidence
                
            return "UNKNOWN", 0.0
            
        except Exception as e:
            print(f"[OCR] Recognition error: {str(e)}")
            return "ERROR", 0.0
    
    def recognize_plates_in_frame(self, frame: np.ndarray, plates: List[Dict]) -> List[Dict]:
        """
        Recognize text from multiple license plate detections in a frame with enhanced processing.
        
        Args:
            frame: Full frame image (BGR format)
            plates: List of plate detections with 'bbox' keys and detection metadata
            
        Returns:
            List of plates with added 'plate_text' and 'plate_confidence' keys
        """
        results = []
        
        for i, plate in enumerate(plates):
            try:
                x1, y1, x2, y2 = map(int, plate['bbox'])
                
                # Add padding to the plate region (10% of width/height)
                h, w = frame.shape[:2]
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                # Skip if plate region is too small
                if x2 <= x1 + 5 or y2 <= y1 + 5:  # At least 5x5 pixels
                    print(f"[OCR] Plate {i}: Region too small or invalid")
                    plate['plate_text'] = "INVALID_REGION"
                    plate['plate_confidence'] = 0.0
                    results.append(plate)
                    continue
                    
                plate_roi = frame[y1:y2, x1:x2]
                
                # Skip if ROI is empty or too small
                if plate_roi.size == 0:
                    print(f"[OCR] Plate {i}: Empty ROI")
                    plate['plate_text'] = "EMPTY_ROI"
                    plate['plate_confidence'] = 0.0
                    results.append(plate)
                    continue
                    
                # Save debug image of the plate ROI
                debug_dir = 'debug_plates'
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f'plate_{i}_debug.jpg')
                cv2.imwrite(debug_path, plate_roi)
                
                # Perform OCR on the plate region
                plate_text, confidence = self.recognize_plate(plate_roi)
                
                # Update plate information
                plate['plate_text'] = plate_text
                plate['plate_confidence'] = float(confidence)
                plate['debug_image'] = debug_path
                
                print(f"[OCR] Plate {i}: Detected '{plate_text}' with confidence {confidence:.2f}")
                
            except Exception as e:
                print(f"[OCR] Error processing plate {i}: {str(e)}")
                plate['plate_text'] = "PROCESSING_ERROR"
                plate['plate_confidence'] = 0.0
                
            results.append(plate)
            
        return results
