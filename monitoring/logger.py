#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitoring/logger.py
================================================================================
سیستم Logging زمان‌محور.
تمام خروجی‌ها از اینجا عبور می‌کنند.
================================================================================
"""

import logging
import os
from constants import LOG_LEVEL, LOG_FILE, LOG_TIMESTAMPS

def setup_logger():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logger = logging.getLogger("climatology")
    logger.setLevel(getattr(logging, LOG_LEVEL))
    if logger.handlers:
        logger.handlers.clear()
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(getattr(logging, LOG_LEVEL))
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, LOG_LEVEL))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s") if LOG_TIMESTAMPS else logging.Formatter("%(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger()
