import math
from typing import Dict, List, Any
from src.core.models import Detection

class RiderAssociationEngine:
    """
    Stateless per-frame spatial association engine.
    Matches persons (riders) to motorcycles using a Center-Point Containment method.
    """

    def __init__(self):
        pass

    def associate(self, routed_detections: Dict[str, List[Detection]]) -> Dict[int, Dict[str, Any]]:
        """
        Perform spatial matching between persons and motorcycles.

        Args:
            routed_detections (Dict[str, List[Detection]]): Output from LogicRouter.

        Returns:
            Dict[int, Dict[str, Any]]: Dictionary keyed by motorcycle track_id mapping to
                "motorcycle": Detection object
                "riders": List of Detection objects (persons associated)
        """
        motorcycles = routed_detections.get("motorcycles", [])
        persons = routed_detections.get("persons", [])

        # Initialize output dictionary
        associations = {}
        for moto in motorcycles:
            if moto.track_id is not None:
                associations[moto.track_id] = {
                    "motorcycle": moto,
                    "riders": []
                }

        # If no motorcycles, return empty (already handled since motorcycles list is empty)
        if not motorcycles:
            return {}

        # If no persons, return motorcycles with empty riders (already done by initialization)
        if not persons:
            return associations

        # Process each person
        for person in persons:
            px, py = person.bbox.center

            # Find all motorcycles whose bounding box contains the person's center
            containing_motos = []
            for moto in motorcycles:
                if moto.track_id is None:
                    continue
                mx1, my1, mx2, my2 = moto.bbox.x1, moto.bbox.y1, moto.bbox.x2, moto.bbox.y2
                # Check point-in-rectangle
                if mx1 <= px <= mx2 and my1 <= py <= my2:
                    containing_motos.append(moto)

            # If person is inside one or more motorcycles, associate with the closest center
            if containing_motos:
                closest_moto = None
                min_distance = float('inf')

                for moto in containing_motos:
                    mx, my = moto.bbox.center
                    dist = math.dist((px, py), (mx, my))

                    if dist < min_distance:
                        min_distance = dist
                        closest_moto = moto

                if closest_moto and closest_moto.track_id is not None:
                    associations[closest_moto.track_id]["riders"].append(person)

        return associations
