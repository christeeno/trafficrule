import cv2
import os
import logging
import time

from src.core.detector import VehicleDetector
from src.core.logic_router import VehicleLogicRouter
from src.utils.drawing import draw_detections

logger = logging.getLogger("TrafficSystem.Pipeline")

class TrafficPipeline:
    """
    Orchestrates the data flow:
    Video Stream -> Vehicle Detection & Tracking -> Annotation -> Video Writer/Display
    """
    def __init__(self, config):
        self.config = config
        
        # Instantiate pure perception layer
        model_cfg = self.config['model']
        self.detector = VehicleDetector(
            model_weight=model_cfg['weights'],
            conf_thresh=model_cfg['confidence_threshold'],
            iou_thresh=model_cfg.get('iou_threshold', 0.45),
            target_classes=model_cfg['target_classes'],
            tracker=model_cfg.get('tracker', 'bytetrack.yaml')
        )
        
        # Instantiate logical routing layer (Phase 2)
        self.logic_router = VehicleLogicRouter()
        
        self.io_cfg = self.config['io']

    def run(self):
        source_path = self.io_cfg['input_source']
        logger.info(f"Starting inference pipeline on source: {source_path}")
        
        # 0 opens webcam, otherwise read string path
        cap = cv2.VideoCapture(int(source_path) if str(source_path).isdigit() else source_path)

        if not cap.isOpened():
            logger.error(f"Cannot initialize video stream from {source_path}")
            return

        out = None
        if self.io_cfg.get('save_results', False):
            out_dir = self.io_cfg.get('output_dir', 'data/output/')
            os.makedirs(out_dir, exist_ok=True)
            
            # Setup Video Writer
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            output_file = os.path.join(out_dir, "phase1_tracked_output.mp4")
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            logger.info(f"Saving output video to: {output_file}")

        frame_count = 0
        processed_count = 0
        start_time = time.time()
        frame_skip = self.io_cfg.get('frame_skip', 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream reached.")
                break

            frame_count += 1
            
            # Frame Skipping Logic 
            # Note: For strict robust multiobject tracking, dropping sequential frames might misalign kalman filter. 
            # In Phase 1.5, we maintain standard skip but rely on bytetrack's robust association.
            if frame_count % frame_skip != 0:
                continue
                
            processed_count += 1

            # 1. Detection & Tracking Layer
            detections = self.detector.detect_and_track(frame)

            # 2. Routing Layer (Phase 2)
            routed_detections = self.logic_router.route(detections)
            logger.info("--- Structured Routing Output ---")
            for category, det_list in routed_detections.items():
                logger.info(f"{category}: {[f'{d.class_name}(ID:{d.track_id})' for d in det_list]}")

            # 3. Annotation Component
            annotated_frame = draw_detections(frame.copy(), detections)

            # Performance & Logging tracker
            if processed_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_calc = processed_count / elapsed
                logger.debug(f"Processing... Frame {frame_count}, Tracked Objects: {len(detections)}, Pipeline FPS: {fps_calc:.1f}")

            # 3. Output Handlers
            if out:
                out.write(annotated_frame)

            if self.io_cfg.get('show_display', True):
                # Resize for display if frame is huge (e.g. 4k)
                disp_frame = cv2.resize(annotated_frame, (1280, 720)) if annotated_frame.shape[1] > 1280 else annotated_frame
                cv2.imshow("Phase 1: Tracked Vehicle Detection", disp_frame)
                
                # Graceful termination
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Pipeline terminated by user.")
                    break

        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline closed successfully.")
