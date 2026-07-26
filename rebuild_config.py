# ============================================================================
# بازنویسی config.yaml (بدون وابستگی به ماژول‌های دیگر)
# ============================================================================

import os

CONFIG_PATH = r"K:\gozareshha\dr vazife\140504 - qc temp\climatology\climatology_engine\config.yaml"

# محتوای سالم YAML
CONFIG_CONTENT = """# تنظیمات اقلیم‌شناسی
years:
  start: 1990
  end: 2020

data:
  zarr_base: "I:/climatology_366_rolling"
  output_zarr: "I:/climatology_366_rolling/climatology_stationwise_final.zarr"

processing:
  block_size: 1000
  max_values_per_fit: 150
  min_valid_values: 10

parallel:
  use_parallel: true
  n_workers: 6
"""

# بازنویسی فایل
with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
    f.write(CONFIG_CONTENT)

print(f"✅ config.yaml بازنویسی شد: {CONFIG_PATH}")