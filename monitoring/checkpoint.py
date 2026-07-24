#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitoring/checkpoint.py
================================================================================
مدیریت Checkpoint به صورت فایل CSV.
شامل: block, station, timestamp, elapsed, version
================================================================================
"""

import os
import time
from constants import CHECKPOINT_FILE

CHECKPOINT_VERSION = 1

def save_checkpoint(block_idx, station_idx, elapsed=None):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(f"block={block_idx}\n")
        f.write(f"station={station_idx}\n")
        f.write(f"timestamp={int(time.time())}\n")
        if elapsed is not None:
            f.write(f"elapsed={elapsed:.1f}\n")
        f.write(f"version={CHECKPOINT_VERSION}\n")

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    data = {}
    with open(CHECKPOINT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                data[key] = value
    if "block" in data and "station" in data:
        return {
            "block": int(data["block"]),
            "station": int(data["station"]),
            "timestamp": int(data.get("timestamp", 0)),
            "elapsed": float(data.get("elapsed", 0)),
            "version": int(data.get("version", 0)),
        }
    return None

def delete_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
