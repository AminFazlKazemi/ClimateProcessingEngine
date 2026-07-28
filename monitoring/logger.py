# -*- coding: utf-8 -*-
"""
Logger configuration for Climatology Engine
"""

import os
import logging
import sys
from datetime import datetime

# Import constants with safe fallback
try:
    from constants import LOG_LEVEL, LOG_FILE, LOG_TIMESTAMPS
except ImportError:
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/climatology.log"
    LOG_TIMESTAMPS = True

def setup_logger():
    """Set up and return the logger instance."""
    # Ensure log directory exists
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger("climatology")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Remove existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    if LOG_TIMESTAMPS:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        formatter = logging.Formatter("%(levelname)s - %(message)s")

    # File handler
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file handler for {LOG_FILE}: {e}")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Create logger instance
logger = setup_logger()
