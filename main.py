import os
from src.utils.logger import setup_logger
from src.config_loader import load_config
from src.core.pipeline import TrafficPipeline

def main():
    # 1. Initialize global system logger
    logger = setup_logger("TrafficSystem", level="DEBUG")
    logger.info("=== Initializing Traffic Violation Detection System (Phase 1) ===")

    # 2. Load constraints and rules
    config = load_config("config.yaml")

    # 3. Mount pipeline with configs
    pipeline = TrafficPipeline(config)

    # 4. Trigger system execution
    pipeline.run()

if __name__ == "__main__":
    main()
