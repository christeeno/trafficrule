import yaml
import os
import logging

logger = logging.getLogger("TrafficSystem.ConfigLoader")

def load_config(config_path="config.yaml"):
    """
    Safely load a YAML configuration file.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
