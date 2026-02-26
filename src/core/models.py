from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class BoundingBox:
    """Formal data structure representing a 2D bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def center(self) -> Tuple[int, int]:
        """Calculates the center (x, y) coordinates of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

@dataclass
class Detection:
    """Standardized entity for any detected frame object (vehicle, person, etc.)."""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None
