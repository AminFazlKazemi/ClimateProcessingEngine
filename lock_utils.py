#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lock_utils.py – مدیریت قفل برای جلوگیری از تداخل کش بین پردازش‌ها
"""

import os
import time
import glob
from pathlib import Path

LOCK_DIR = Path("./locks")
LOCK_DIR.mkdir(exist_ok=True)

def get_lock_path(block_idx):
    """مسیر فایل قفل برای یک بلوک"""
    return LOCK_DIR / f"lock_{block_idx:04d}.lock"

def acquire_lock(block_idx, timeout=5):
    """
    تلاش برای گرفتن قفل یک بلوک.
    اگر قفل در دسترس بود، True برمی‌گرداند.
    اگر timeout تمام شد، False برمی‌گرداند.
    """
    lock_path = get_lock_path(block_idx)
    start_time = time.time()
    
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.5)

def release_lock(block_idx):
    """آزاد کردن قفل یک بلوک"""
    lock_path = get_lock_path(block_idx)
    if lock_path.exists():
        lock_path.unlink()

def is_locked(block_idx):
    """بررسی اینکه آیا یک بلوک قفل است"""
    return get_lock_path(block_idx).exists()

def clean_stale_locks(timeout_seconds=300):
    """
    پاک کردن قفل‌های قدیمی (بیش از timeout_seconds ثانیه).
    این کار از باقی ماندن قفل‌های مرده جلوگیری می‌کند.
    """
    now = time.time()
    for lock_file in LOCK_DIR.glob("*.lock"):
        if now - lock_file.stat().st_mtime > timeout_seconds:
            lock_file.unlink()
            print(f"🗑️ قفل قدیمی پاک شد: {lock_file.name}")