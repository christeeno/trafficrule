import logging
from typing import List, Dict
from src.core.models import Detection
from src.utils.logger import setup_logger

logger = setup_logger("TrafficSystem.LogicRouter")

class VehicleLogicRouter:
    """
    Conditional logic layer for routing vehicle detections to future modules.
    Categorizes vehicles by type and groups them logically.
    Maintains strict separation of concerns and operates per frame (stateless).
    """

    def __init__(self):
        pass

    def route(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """
        Receives list of Detection objects from Phase 1 and routes them to logical pipelines.
        
        Args:
            detections (List[Detection]): List of Detection dataclasses.
            
        Returns:
            Dict[str, List[Detection]]: Structured routing output grouped by vehicle type.
        """
        routed_data: Dict[str, List[Detection]] = {
            "motorcycles": [],
            "cars": [],
            "heavy_vehicles": []
        }

        for det in detections:
            cat_name = det.class_name.lower()
            if cat_name == "motorcycle":
                routed_data["motorcycles"].append(det)
            elif cat_name == "car":
                routed_data["cars"].append(det)
            elif cat_name in ["bus", "truck"]:
                routed_data["heavy_vehicles"].append(det)
            else:
                logger.debug(f"Vehicle class '{cat_name}' not mapped in LogicRouter.")

        return routed_data
