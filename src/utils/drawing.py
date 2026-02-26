import cv2

# BGR Color map for visibility
COLOR_MAP = {
    0: (255, 105, 180), # Person (Wait for Phase 2 constraint) - Hot Pink
    2: (255, 0, 0),    # Car - Blue
    3: (0, 0, 255),    # Motorcycle - Red
    5: (0, 255, 0),    # Bus - Green
    7: (0, 255, 255)   # Truck - Yellow
}

def draw_detections(frame, detections):
    """
    Draws bounding boxes, track IDs, and confidence scores over detected vehicles.
    Decoupled from inference logic for cleaner architecture.
    """
    for det in detections:
        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
        cls_id = det.class_id
        cls_name = det.class_name.capitalize()
        conf = det.confidence
        track_id = det.track_id

        # Determine color based on class ID or fallback to white
        color = COLOR_MAP.get(cls_id, (255, 255, 255))

        # Draw Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Structure label with persistent tracker ID
        track_str = f"ID:{track_id} " if track_id is not None else ""
        label = f"{track_str}{cls_name} {conf:.2f}"
        
        # Calculate text background size
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        
        # Draw Label Background
        cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)
        
        # Draw Label Text
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame
