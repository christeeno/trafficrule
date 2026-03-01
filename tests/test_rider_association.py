import pytest
from src.core.models import Detection, BoundingBox
from src.core.rider_association import RiderAssociationEngine

@pytest.fixture
def engine():
    return RiderAssociationEngine()

def create_person(track_id, center_x, center_y, width=20, height=40):
    """Helper to create a person detection center around (center_x, center_y)"""
    x1, y1 = center_x - width // 2, center_y - height // 2
    x2, y2 = center_x + width // 2, center_y + height // 2
    return Detection(
        class_id=0, class_name="person", confidence=0.9, 
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), track_id=track_id
    )

def create_moto(track_id, x1, y1, x2, y2):
    return Detection(
        class_id=1, class_name="motorcycle", confidence=0.8,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), track_id=track_id
    )

def test_single_motorcycle_single_rider(engine):
    moto = create_moto(track_id=10, x1=100, y1=100, x2=200, y2=200)
    # Rider center inside moto bbox (center: 150, 150)
    rider = create_person(track_id=1, center_x=150, center_y=150)
    
    routed_detections = {
        "motorcycles": [moto],
        "persons": [rider],
        "cars": [], "heavy_vehicles": []
    }
    
    result = engine.associate(routed_detections)
    
    assert 10 in result
    assert result[10]["motorcycle"] == moto
    assert len(result[10]["riders"]) == 1
    assert result[10]["riders"][0] == rider

def test_single_motorcycle_two_riders(engine):
    moto = create_moto(track_id=10, x1=100, y1=100, x2=200, y2=200)
    # Two riders inside moto bbox
    rider1 = create_person(track_id=1, center_x=130, center_y=130)
    rider2 = create_person(track_id=2, center_x=170, center_y=170)
    
    routed_detections = {
        "motorcycles": [moto],
        "persons": [rider1, rider2]
    }
    
    result = engine.associate(routed_detections)
    assert len(result[10]["riders"]) == 2
    assert rider1 in result[10]["riders"]
    assert rider2 in result[10]["riders"]

def test_single_motorcycle_three_riders(engine):
    moto = create_moto(track_id=10, x1=100, y1=100, x2=200, y2=200)
    # Three riders inside moto bbox
    rider1 = create_person(track_id=1, center_x=120, center_y=120)
    rider2 = create_person(track_id=2, center_x=150, center_y=150)
    rider3 = create_person(track_id=3, center_x=180, center_y=180)
    
    routed_detections = {
        "motorcycles": [moto],
        "persons": [rider1, rider2, rider3]
    }
    
    result = engine.associate(routed_detections)
    assert len(result[10]["riders"]) == 3

def test_motorcycle_with_no_riders(engine):
    moto = create_moto(track_id=10, x1=100, y1=100, x2=200, y2=200)
    
    routed_detections = {
        "motorcycles": [moto],
        "persons": []
    }
    
    result = engine.associate(routed_detections)
    assert len(result) == 1
    assert len(result[10]["riders"]) == 0

def test_persons_not_overlapping_any_motorcycle(engine):
    moto = create_moto(track_id=10, x1=100, y1=100, x2=200, y2=200)
    # Rider completely outside moto bbox
    rider1 = create_person(track_id=1, center_x=50, center_y=50) # outside top-left
    rider2 = create_person(track_id=2, center_x=300, center_y=300) # outside bottom-right
    
    routed_detections = {
        "motorcycles": [moto],
        "persons": [rider1, rider2]
    }
    
    result = engine.associate(routed_detections)
    assert len(result[10]["riders"]) == 0

def test_multiple_motorcycles_with_distinct_riders(engine):
    moto1 = create_moto(track_id=10, x1=100, y1=100, x2=200, y2=200) # Center: 150, 150
    moto2 = create_moto(track_id=20, x1=300, y1=300, x2=400, y2=400) # Center: 350, 350
    
    rider1 = create_person(track_id=1, center_x=150, center_y=150) # in moto1
    rider2 = create_person(track_id=2, center_x=350, center_y=350) # in moto2
    
    routed_detections = {
        "motorcycles": [moto1, moto2],
        "persons": [rider1, rider2]
    }
    
    result = engine.associate(routed_detections)
    assert len(result) == 2
    assert len(result[10]["riders"]) == 1
    assert result[10]["riders"][0] == rider1
    assert len(result[20]["riders"]) == 1
    assert result[20]["riders"][0] == rider2

def test_person_overlapping_multiple_motorcycles(engine):
    # Two motorcycles overlapping
    moto1 = create_moto(track_id=10, x1=100, y1=100, x2=200, y2=200) # center 150, 150
    moto2 = create_moto(track_id=20, x1=150, y1=100, x2=250, y2=200) # center 200, 150
    
    # Rider closer to moto2 center (200, 150)
    rider = create_person(track_id=1, center_x=190, center_y=150)
    
    routed_detections = {
        "motorcycles": [moto1, moto2],
        "persons": [rider]
    }
    
    result = engine.associate(routed_detections)
    # Rider falls inside both bounding boxes.
    # Dist to moto1 center (150, 150): 40
    # Dist to moto2 center (200, 150): 10
    # Should be associated with moto2.
    assert len(result[20]["riders"]) == 1
    assert result[20]["riders"][0] == rider
    assert len(result[10]["riders"]) == 0

def test_stateless_multi_frame(engine):
    # Frame 1
    moto = create_moto(track_id=10, x1=100, y1=100, x2=200, y2=200)
    rider = create_person(track_id=1, center_x=150, center_y=150)
    res_f1 = engine.associate({"motorcycles": [moto], "persons": [rider]})
    assert len(res_f1[10]["riders"]) == 1
    
    # Frame 2 (no motorcycles or persons)
    res_f2 = engine.associate({"motorcycles": [], "persons": []})
    assert len(res_f2) == 0
    
    # Frame 3 (rider moved outside)
    moto = create_moto(track_id=10, x1=100, y1=100, x2=200, y2=200)
    rider = create_person(track_id=1, center_x=300, center_y=300)
    res_f3 = engine.associate({"motorcycles": [moto], "persons": [rider]})
    assert len(res_f3[10]["riders"]) == 0
