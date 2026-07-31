#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checkpoint_manager.py – مدیریت یکپارچه‌ی فایل checkpoint با تشخیص خودکار
================================================================================
- بررسی وجود فایل checkpoint در مسیر صحیح
- ایجاد خودکار فایل با مقادیر پیش‌فرض در صورت نبود
- تشخیص خودکار آخرین نقطه‌ی معتبر از Zarr و تنظیم checkpoint بر اساس آن
================================================================================
"""

import os
import time
import numpy as np
import xarray as xr
from constants import CHECKPOINT_FILE, OUTPUT_DIR, OUTPUT_ZARR, BLOCK_SIZE, N_DAYS


def ensure_checkpoint_dir():
    """اطمینان از وجود پوشه‌ی والد فایل checkpoint"""
    dirname = os.path.dirname(CHECKPOINT_FILE)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        print(f"📁 پوشه‌ی checkpoint ساخته شد: {dirname}")


def get_default_checkpoint():
    """برگرداندن دیکشنری پیش‌فرض checkpoint (شروع از ابتدا)"""
    return {
        "block": 0,
        "station": 0,
        "timestamp": int(time.time()),
        "version": 1,
    }


def load_checkpoint_dict():
    """
    بارگذاری فایل checkpoint و برگرداندن دیکشنری.
    اگر فایل وجود نداشت یا خراب بود، None برمی‌گرداند.
    """
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    data = {}
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    data[key] = value

        return {
            "block": int(data.get("block", 0)),
            "station": int(data.get("station", 0)),
            "timestamp": int(data.get("timestamp", 0)),
            "version": int(data.get("version", 1)),
        }
    except Exception as e:
        print(f"⚠️ خطا در خواندن checkpoint: {e}")
        return None


def save_checkpoint_dict(checkpoint):
    """ذخیره‌ی دیکشنری checkpoint در فایل."""
    ensure_checkpoint_dir()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        f.write(f"block={checkpoint['block']}\n")
        f.write(f"station={checkpoint['station']}\n")
        f.write(f"timestamp={checkpoint.get('timestamp', int(time.time()))}\n")
        f.write(f"version={checkpoint.get('version', 1)}\n")
        f.flush()
        os.fsync(f.fileno())


def save_checkpoint(block, station):
    """ذخیره‌ی سریع checkpoint با بلوک و ایستگاه مشخص"""
    checkpoint = {
        "block": block,
        "station": station,
        "timestamp": int(time.time()),
        "version": 1,
    }
    save_checkpoint_dict(checkpoint)


def load_checkpoint():
    """بارگذاری checkpoint و برگرداندن دیکشنری (با fallback به پیش‌فرض)"""
    cp = load_checkpoint_dict()
    if cp is None:
        return get_default_checkpoint()
    return cp


def reset_checkpoint(block=0, station=0):
    """بازنشانی checkpoint به مقادیر مشخص"""
    checkpoint = {
        "block": block,
        "station": station,
        "timestamp": int(time.time()),
        "version": 1,
    }
    save_checkpoint_dict(checkpoint)
    print(f"🔄 checkpoint بازنشانی شد: بلوک {block}, ایستگاه {station}")
    return checkpoint


def delete_checkpoint():
    """حذف فایل checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"🗑️ فایل checkpoint حذف شد: {CHECKPOINT_FILE}")


def auto_detect_last_valid_point(zarr_path=None, var_name="tmean_mean", day_idx=-1):
    """
    تشخیص خودکار آخرین نقطه‌ی معتبر در فایل Zarr.
    
    پارامترها:
        zarr_path: مسیر فایل Zarr (پیش‌فرض: OUTPUT_ZARR)
        var_name: نام متغیر برای بررسی (پیش‌فرض: tmean_mean)
        day_idx: اندیس روز (پیش‌فرض: -1 یعنی آخرین روز)
    
    بازگشت:
        دیکشنری شامل:
            - n_valid: تعداد نقاط معتبر
            - last_block: آخرین بلوک کامل
            - last_station: آخرین ایستگاه معتبر
            - checkpoint: دیکشنری checkpoint پیشنهادی
    """
    if zarr_path is None:
        zarr_path = OUTPUT_ZARR
    
    # اگر Zarr وجود ندارد، از ابتدا شروع کن
    if not os.path.exists(zarr_path):
        print(f"⚠️ فایل Zarr وجود ندارد: {zarr_path}")
        print("🆕 شروع از ابتدا (بلوک ۰، ایستگاه ۰)")
        return {
            "n_valid": 0,
            "last_block": 0,
            "last_station": 0,
            "checkpoint": get_default_checkpoint()
        }
    
    try:
        # باز کردن Zarr
        ds = xr.open_zarr(zarr_path, consolidated=False)
        
        # بررسی وجود متغیر
        if var_name not in ds:
            print(f"⚠️ متغیر {var_name} در Zarr وجود ندارد")
            ds.close()
            return {
                "n_valid": 0,
                "last_block": 0,
                "last_station": 0,
                "checkpoint": get_default_checkpoint()
            }
        
        # ✅ اصلاح: استفاده از ds.sizes به جای ds.dims (رفع FutureWarning)
        if day_idx < 0:
            day_idx = ds.sizes['day'] - 1
        
        data = ds[var_name].isel(day=day_idx).values
        
        # تعداد نقاط معتبر
        valid_mask = ~np.isnan(data)
        n_valid = np.sum(valid_mask)
        
        ds.close()
        
        # محاسبه بلوک و ایستگاه
        if n_valid == 0:
            last_block = 0
            last_station = 0
        else:
            # آخرین ایستگاه معتبر
            last_station = np.max(np.where(valid_mask)[0])
            last_block = last_station // BLOCK_SIZE
        
        # checkpoint پیشنهادی
        checkpoint = {
            "block": last_block,
            "station": (last_block + 1) * BLOCK_SIZE if last_block > 0 else 0,
            "timestamp": int(time.time()),
            "version": 1,
        }
        
        print(f"📊 تشخیص خودکار از Zarr:")
        print(f"   تعداد نقاط معتبر در روز {day_idx}: {n_valid:,}")
        print(f"   آخرین ایستگاه معتبر: {last_station:,}")
        print(f"   آخرین بلوک کامل: {last_block}")
        print(f"   نقطه‌ی شروع پیشنهادی: بلوک {checkpoint['block']}, ایستگاه {checkpoint['station']}")
        
        return {
            "n_valid": n_valid,
            "last_block": last_block,
            "last_station": last_station,
            "checkpoint": checkpoint
        }
        
    except Exception as e:
        print(f"⚠️ خطا در تشخیص خودکار: {e}")
        return {
            "n_valid": 0,
            "last_block": 0,
            "last_station": 0,
            "checkpoint": get_default_checkpoint()
        }


def ensure_checkpoint(auto_detect=False):
    """
    اطمینان از وجود فایل checkpoint با تشخیص خودکار.
    
    پارامترها:
        auto_detect: اگر True باشد، موقعیت از Zarr تشخیص داده می‌شود.
    """
    ensure_checkpoint_dir()
    
    # اگر تشخیص خودکار فعال باشد
    if auto_detect:
        detection = auto_detect_last_valid_point()
        if detection["n_valid"] > 0:
            # checkpoint را با مقدار تشخیص‌شده به‌روز کن
            checkpoint = detection["checkpoint"]
            save_checkpoint_dict(checkpoint)
            print(f"✅ checkpoint به‌روز شد: بلوک {checkpoint['block']}, ایستگاه {checkpoint['station']}")
            return checkpoint
    
    # اگر فایل وجود ندارد
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"🆕 هیچ checkpointی وجود ندارد. ایجاد checkpoint پیش‌فرض (شروع از ابتدا)...")
        checkpoint = get_default_checkpoint()
        save_checkpoint_dict(checkpoint)
        return checkpoint
    
    # بارگذاری فایل موجود
    checkpoint = load_checkpoint_dict()
    if checkpoint is None:
        print("⚠️ فایل checkpoint خراب است. بازنشانی به مقادیر پیش‌فرض...")
        checkpoint = get_default_checkpoint()
        save_checkpoint_dict(checkpoint)
    
    return checkpoint


def print_checkpoint_status():
    """چاپ وضعیت فعلی checkpoint"""
    cp = ensure_checkpoint()
    print(f"📍 وضعیت checkpoint: بلوک {cp['block']}, ایستگاه {cp['station']}")
    return cp


# ============================================================
# تست
# ============================================================
if __name__ == "__main__":
    print("🧪 تست checkpoint_manager با تشخیص خودکار...")
    print(f"مسیر checkpoint: {CHECKPOINT_FILE}")
    print(f"مسیر Zarr: {OUTPUT_ZARR}")
    
    # تست تشخیص خودکار
    cp = ensure_checkpoint(auto_detect=True)
    print(f"checkpoint نهایی: {cp}")
    
    print("\n✅ تست کامل شد.")