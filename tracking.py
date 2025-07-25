import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO

class Tracker:
    """Wrapper around Ultralytics YOLOv8 tracker (ByteTrack/StrongSORT).
    It returns an annotated frame and a list of detections containing
    bounding box, confidence, class id and a persistent track id.
    """
    def __init__(self, model_path: str):
        # The same model used for detection is reused for tracking.
        self.model = YOLO(model_path)

    def track(self, frame: np.ndarray, persist: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Run tracking on a single frame.

        Args:
            frame: BGR image (np.ndarray)
            persist: Keep track IDs across subsequent calls.

        Returns:
            annotated_frame: frame with drawn boxes, ids, etc.
            detections: list of dicts with keys [id, bbox, conf, cls]
        """
        results = self.model.track(frame, persist=persist, verbose=False, conf=0.25, iou=0.5)

        annotated_frame = results[0].plot()
        detections: List[Dict] = []
        for box in results[0].boxes:
            det = {
                "id": int(box.id[0]) if box.id is not None else -1,
                "bbox": list(map(int, box.xyxy[0].tolist())),
                "conf": float(box.conf[0]),
                "cls": int(box.cls[0])
            }
            detections.append(det)
        return annotated_frame, detections
