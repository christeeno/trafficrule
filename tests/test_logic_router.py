import pytest
import copy
from typing import List

from src.core.models import Detection, BoundingBox
from src.core.logic_router import VehicleLogicRouter

@pytest.fixture
def logic_router():
    """Provides a fresh, stateless instance of the VehicleLogicRouter for testing."""
    return VehicleLogicRouter()

def create_mock_detection(cid: int, cname: str, tid: int, conf: float = 0.9) -> Detection:
    """Helper function to quickly generate standardized mock Detection objects."""
    return Detection(
        class_id=cid,
        class_name=cname,
        confidence=conf,
        bbox=BoundingBox(x1=10, y1=10, x2=100, y2=100),
        track_id=tid
    )

class TestVehicleLogicRouterUnit:
    """Unit test suite for Phase 2: Conditional Logic Router."""

    def test_mixed_vehicle_types(self, logic_router):
        """1. Verify correct categorization of mixed vehicle types."""
        detections = [
            create_mock_detection(0, "person", 100),
            create_mock_detection(1, "motorcycle", 101),
            create_mock_detection(2, "car", 102),
            create_mock_detection(3, "bus", 103),
            create_mock_detection(4, "truck", 104)
        ]

        result = logic_router.route(detections)

        assert len(result["persons"]) == 1
        assert result["persons"][0].track_id == 100

        assert len(result["motorcycles"]) == 1
        assert result["motorcycles"][0].track_id == 101
        
        assert len(result["cars"]) == 1
        assert result["cars"][0].track_id == 102
        
        assert len(result["heavy_vehicles"]) == 2
        heavy_ids = [d.track_id for d in result["heavy_vehicles"]]
        assert 103 in heavy_ids
        assert 104 in heavy_ids

    def test_single_category_only(self, logic_router):
        """2. Verify behavior when only one category is present."""
        detections = [
            create_mock_detection(1, "motorcycle", 201),
            create_mock_detection(1, "motorcycle", 202)
        ]

        result = logic_router.route(detections)

        assert len(result["motorcycles"]) == 2
        assert "persons" in result and len(result["persons"]) == 0
        assert "cars" in result and len(result["cars"]) == 0
        assert "heavy_vehicles" in result and len(result["heavy_vehicles"]) == 0

    def test_empty_input_list(self, logic_router):
        """3. Verify router does not crash on empty input and returns proper empty schema."""
        detections: List[Detection] = []

        result = logic_router.route(detections)

        assert isinstance(result, dict)
        assert len(result["persons"]) == 0
        assert len(result["motorcycles"]) == 0
        assert len(result["cars"]) == 0
        assert len(result["heavy_vehicles"]) == 0

    def test_unknown_class(self, logic_router):
        """4. Verify router safely ignores unknown classes without crashing."""
        detections = [
            create_mock_detection(5, "bicycle", 301),
            create_mock_detection(2, "car", 302)
        ]

        result = logic_router.route(detections)

        # "bicycle" should be dropped/ignored
        assert len(result["persons"]) == 0
        assert len(result["motorcycles"]) == 0
        assert len(result["heavy_vehicles"]) == 0
        assert len(result["cars"]) == 1
        assert result["cars"][0].track_id == 302

    def test_duplicate_track_ids(self, logic_router):
        """5. Verify router handles duplicate track_ids gracefully (no modification)."""
        detections = [
            create_mock_detection(2, "car", 999),
            create_mock_detection(2, "car", 999)
        ]

        result = logic_router.route(detections)

        assert len(result["cars"]) == 2
        assert result["cars"][0].track_id == 999
        assert result["cars"][1].track_id == 999

    def test_data_integrity(self, logic_router):
        """6. Verify router does not mutate original Detection objects."""
        original_car = create_mock_detection(2, "car", 501, conf=0.88)
        detections = [original_car]
        
        # Deep copy to ensure we have a perfect baseline to compare against
        baseline = copy.deepcopy(original_car)

        result = logic_router.route(detections)
        routed_car = result["cars"][0]

        assert routed_car.class_id == baseline.class_id
        assert routed_car.class_name == baseline.class_name
        assert routed_car.confidence == baseline.confidence
        assert routed_car.track_id == baseline.track_id
        assert routed_car.bbox.x1 == baseline.bbox.x1
        assert routed_car.bbox.y2 == baseline.bbox.y2

class TestVehicleLogicRouterIntegration:
    """Integration style tests ensuring structural contract logic over multiple simulated frames."""

    def test_stateless_multi_frame_routing(self, logic_router):
        """Verify the router maintains no state across sequential frames."""
        # Frame 1: 1 Car
        frame_1_dets = [create_mock_detection(2, "car", 1)]
        res_1 = logic_router.route(frame_1_dets)
        assert len(res_1["cars"]) == 1
        assert len(res_1["motorcycles"]) == 0

        # Frame 2: 1 Motorcycle
        frame_2_dets = [create_mock_detection(1, "motorcycle", 2)]
        res_2 = logic_router.route(frame_2_dets)
        assert len(res_2["cars"]) == 0           # Previously detected car shouldn't persist
        assert len(res_2["motorcycles"]) == 1

        # Frame 3: Empty
        res_3 = logic_router.route([])
        assert len(res_3["persons"]) == 0
        assert len(res_3["cars"]) == 0
        assert len(res_3["motorcycles"]) == 0
