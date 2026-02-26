import logging
from ultralytics import YOLO
from src.core.models import Detection, BoundingBox

logger = logging.getLogger("TrafficSystem.Detector")

class VehicleDetector:
    """
    Wraps the YOLOv8 model for pure perception.
    Responsible exclusively for detecting and tracking objects statelessly per frame.
    Tracking state is handled natively by YOLO with persist=True.
    """
    def __init__(self, model_weight, conf_thresh, iou_thresh, target_classes, tracker):
        logger.info(f"Initializing YOLO Model with weights: {model_weight}")
        self.model = YOLO(model_weight)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_classes = target_classes
        self.tracker_config = tracker
        
        logger.debug(f"Detector Filters -> Conf: {conf_thresh}, Classes: {target_classes}")

    def detect_and_track(self, frame):
        """
        Runs object detection and multiobject tracking on a single frame.
        
        Returns:
            list of Detection: Standardized detection dataclasses
        """
        # verbose=False prevents YOLO from cluttering the console output on every frame
        results = self.model.track(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.target_classes,
            tracker=self.tracker_config,
            persist=True,
            verbose=False
        )

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Standard Python int/float mapping ensures downstream systems don't need torch/ultralytics imports
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Fetch tracking ID if available
                track_id = int(box.id[0]) if box.id is not None else None

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    track_id=track_id
                ))

        return detections
