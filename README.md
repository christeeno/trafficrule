# AI-Based Traffic Violation Detection & Civic Analytics System

**Lead Developer:** Christeeno Telfin  
**Project Status:** Phase 3 (Rider Association) in Development

---

## 1. Executive Summary

This project is a modular, high-performance computer vision framework designed for real-time traffic monitoring and automated enforcement. Unlike traditional monolithic detection scripts, this system utilizes a decoupled, stateless architecture to identify traffic violations—such as helmet non-compliance and triple riding—while generating behavioral analytics for smart city infrastructure.

## 2. The Challenge

Current urban traffic enforcement is hindered by:

- **Scalability Gaps:** Manual monitoring cannot realistically cover expanding road networks.
- **Delayed Intervention:** Human-in-the-loop reporting creates a significant lag between the violation and enforcement.
- **Data Silos:** Existing CCTV systems often fail to convert raw video into actionable urban planning intelligence.

## 3. System Architecture & Pipeline

The system follows a "Perception-to-Policy" pipeline, ensuring that AI model internals remain isolated from the business logic.

### Phase 1: Perception Layer

- **Inference Engine:** YOLOv8 for high-speed object detection.
- **Tracking:** ByteTrack integration to maintain multi-object persistence across frames.
- **Data Standard:** Standardized output via custom Python dataclasses for model-agnostic downstream processing.

### Phase 2: Stateless Logic Router

A specialized routing layer that categorizes detections (e.g., Motorcycles, Pedestrians, Heavy Vehicles) without maintaining internal state.

**Benefit:** Enables rigorous unit testing and ensures the system can be seamlessly scaled across distributed edge devices.

### Phase 3: Rider Association (Active Development)

- **Spatial Heuristics:** Algorithmic association of "Person" entities to "Motorcycle" bounding boxes.
- **Violation Logic:** Utilizes spatial overlaps and centroid proximity to detect Triple Riding events automatically.

## 4. Automated Enforcement Workflow

Upon identifying a violation, the system generates a structured, machine-verifiable Evidence Packet.

### Proposed Data Structure (JSON)

```json
{
  "violation_id": "VR-2026-X892",
  "type": "Triple_Riding",
  "confidence_score": 0.94,
  "telemetry": {
    "timestamp": "2026-02-26T20:35:00Z",
    "location_geo": [10.352, 76.213],
    "lane_id": "NH-544-South"
  },
  "evidence": {
    "license_plate": "KL-45-Q-1234",
    "frame_path": "storage/events/EV_001.jpg"
  }
}
```

## 5. Future Scope: Civic Analytics & Urban Intelligence

The long-term vision is to transition from an enforcement tool into a comprehensive Civic Intelligence Layer.

### A. Heatmap-Based Traffic Behavior Intelligence

By aggregating geospatial data from violation events, the system will generate:

- **Violation Density Heatmaps:** Identifying high-risk zones for targeted enforcement.
- **Compliance Mapping:** Visualizing regional trends in helmet and seatbelt usage.
- **Risk Intersections:** Detecting patterns in lane misuse and night-time violations.

### B. Smart City Planning & Analytics

This data enables urban planners to make evidence-based decisions regarding infrastructure and safety.

| Feature | Insight Provided | Application |
|---------|-----------------|------------|
| Compliance Heatmaps | Geofenced violation density | Targeted police deployment |
| Class Distribution | Ratio of heavy vehicles vs. commuters | Road wear-and-tear forecasting |
| Peak-Hour Analytics | Temporal violation trends | Dynamic traffic signal timing |

---

## Project Structure

```
trafficrule/
├── main.py              # Entry point
├── config.yaml          # Configuration file
├── requirements.txt     # Project dependencies
├── yolov8n.pt          # YOLOv8 model weights
├── data/
│   ├── input/          # Input video/image data
│   └── output/         # Processing results
├── src/
│   ├── config_loader.py    # Configuration management
│   ├── core/
│   │   ├── detector.py     # YOLOv8 inference engine
│   │   ├── logic_router.py # Stateless logic routing
│   │   ├── models.py       # Data models
│   │   └── pipeline.py     # Processing pipeline
│   └── utils/
│       ├── drawing.py      # Visualization utilities
│       └── logger.py       # Logging utilities
└── tests/
    └── test_logic_router.py # Unit tests
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone or download the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure model weights are available:
   ```
   yolov8n.pt (included in repository)
   ```

### Configuration

Edit `config.yaml` to configure:
- Model parameters
- Detection thresholds
- Output paths
- Logging settings

### Running the System

```bash
python main.py
```

## Testing

Run unit tests with:
```bash
pytest tests/
```

---

**Note:** This project is under active development. Features and APIs may change during Phase 3 implementation.
