from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from ultralytics import YOLO

class TruckClassifier:
    def __init__(self, min_plate_distance: float = 0.3, max_plate_distance: float = 0.8, model_path: str = 'yolov8s.pt'):
        """
        Initialize the truck classifier.
        
        Args:
            min_plate_distance: Minimum normalized distance between plates to consider them as separate
            max_plate_distance: Maximum normalized distance between plates to consider them as belonging to the same truck
            model_path: Path to the YOLO model for truck classification
        """
        self.min_plate_distance = min_plate_distance
        self.max_plate_distance = max_plate_distance
        self.plate_pairs = {}  # Track plate pairs for each truck
        self.model = YOLO(model_path) if model_path else None
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        # Determine the coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate area of intersection
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        
        # Calculate area of both boxes
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        
        # Calculate IoU
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def classify(self, frame: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
        """
        Classify a truck based on its visual features.
        
        Args:
            frame: Input BGR image
            bbox: Bounding box [x1, y1, x2, y2] of the truck
            
        Returns:
            Dictionary containing classification results:
            - 'class_name': Predicted class (e.g., 'truck', 'trailer')
            - 'confidence': Confidence score (0-1)
            - 'features': Extracted features (if any)
        """
        try:
            # Extract the truck region
            x1, y1, x2, y2 = map(int, bbox)
            truck_roi = frame[max(0, y1):min(frame.shape[0], y2), 
                            max(0, x1):min(frame.shape[1], x2)]
            
            if truck_roi.size == 0:
                return {'class_name': 'unknown', 'confidence': 0.0, 'features': None}
                
            # Simple classification based on aspect ratio
            height, width = truck_roi.shape[:2]
            aspect_ratio = width / max(height, 1)
            
            # Basic classification rules
            if aspect_ratio > 2.5:
                return {'class_name': 'truck_with_trailer', 'confidence': 0.8, 'features': None}
            elif aspect_ratio > 1.5:
                return {'class_name': 'truck', 'confidence': 0.9, 'features': None}
            else:
                return {'class_name': 'van', 'confidence': 0.7, 'features': None}
                
        except Exception as e:
            print(f"[ERROR] Classification failed: {str(e)}")
            return {'class_name': 'unknown', 'confidence': 0.0, 'features': None}
    
    def _calculate_plate_distance(self, truck_bbox: List[int], plate_bbox: List[int]) -> float:
        """
        Calculate normalized distance between truck and plate centers.
        
        Returns:
            Normalized distance (0-1) where 0 means centers are the same and 1 means maximum possible distance
        """
        # Calculate centers
        truck_cx = (truck_bbox[0] + truck_bbox[2]) / 2
        truck_cy = (truck_bbox[1] + truck_bbox[3]) / 2
        plate_cx = (plate_bbox[0] + plate_bbox[2]) / 2
        plate_cy = (plate_bbox[1] + plate_bbox[3]) / 2
        
        # Calculate Euclidean distance between centers
        distance = np.sqrt((truck_cx - plate_cx)**2 + (truck_cy - plate_cy)**2)
        
        # Normalize by truck diagonal
        truck_diag = np.sqrt((truck_bbox[2] - truck_bbox[0])**2 + (truck_bbox[3] - truck_bbox[1])**2)
        normalized_distance = distance / truck_diag if truck_diag > 0 else 1.0
        
        return normalized_distance
    
    def _assign_plates_to_trucks(self, trucks: List[Dict], plates: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Assign license plates to the nearest truck.
        
        Returns:
            Dictionary mapping truck index to list of associated plates
        """
        truck_plates = {i: [] for i in range(len(trucks))}
        
        for plate in plates:
            plate_bbox = plate['bbox']
            min_distance = float('inf')
            best_truck_idx = -1
            
            # Find the closest truck to this plate
            for i, truck in enumerate(trucks):
                truck_bbox = truck['bbox']
                distance = self._calculate_plate_distance(truck_bbox, plate_bbox)
                
                # Only consider plates that are reasonably close to a truck
                if distance < self.max_plate_distance and distance < min_distance:
                    min_distance = distance
                    best_truck_idx = i
            
            if best_truck_idx != -1:
                truck_plates[best_truck_idx].append(plate)
        
        return truck_plates
    
    def classify_trucks(self, trucks: List[Dict], plates: List[Dict]) -> List[Dict]:
        """
        Classify trucks as single or double/articulated based on license plates.
        
        Args:
            trucks: List of truck detections
            plates: List of plate detections with text recognition results
            
        Returns:
            List of trucks with added 'classification' and 'plates' keys
        """
        if not trucks:
            return []
        
        # Assign plates to trucks
        truck_plates = self._assign_plates_to_trucks(trucks, plates)
        
        results = []
        
        for i, truck in enumerate(trucks):
            plates_for_truck = truck_plates.get(i, [])
            
            # Update truck with plates
            truck['plates'] = plates_for_truck
            
            # Classify based on number of plates
            if len(plates_for_truck) == 0:
                truck['classification'] = 'unknown'
                truck['classification_confidence'] = 0.0
            elif len(plates_for_truck) == 1:
                truck['classification'] = 'single'
                truck['classification_confidence'] = plates_for_truck[0]['confidence']
            else:
                # Check if plates are far enough apart to be front/back
                plate_positions = []
                for plate in plates_for_truck:
                    # Calculate normalized y-position of plate (0=top, 1=bottom of truck)
                    plate_y = (plate['bbox'][1] + plate['bbox'][3]) / 2
                    truck_y1, truck_y2 = truck['bbox'][1], truck['bbox'][3]
                    normalized_y = (plate_y - truck_y1) / (truck_y2 - truck_y1) if truck_y2 > truck_y1 else 0.5
                    plate_positions.append(normalized_y)
                
                # Check if plates are at opposite ends of the truck
                plate_positions.sort()
                plate_distance = plate_positions[-1] - plate_positions[0]
                
                if plate_distance > self.min_plate_distance and len(plates_for_truck) == 2:
                    # Two plates at opposite ends - likely front and back
                    truck['classification'] = 'double/articulated'
                    truck['classification_confidence'] = min(p['confidence'] for p in plates_for_truck)
                else:
                    # Multiple plates but too close together - probably false detections
                    truck['classification'] = 'single'
                    truck['classification_confidence'] = max(p['confidence'] for p in plates_for_truck)
            
            results.append(truck)
        
        return results
