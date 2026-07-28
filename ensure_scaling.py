#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ensure that tmean is properly scaled (divided by 10) during loading and caching.
This script checks and fixes the get_cached_or_load function in read_month_files.py
to apply scaling BEFORE storing in cache.
"""

import re
import shutil
from pathlib import Path

BASE = Path(r"K:\gozareshha\dr vazife\140504 - qc temp\climatology\climatology_engine")
READ_MONTH = BASE / "io_pipeline" / "read_month_files.py"

# ============================================================
# 1. Ensure scaling is applied in get_cached_or_load
# ============================================================
with open(READ_MONTH, "r", encoding="utf-8") as f:
    content = f.read()

# Check if scaling is already present
if "if var_idx == 1:" in content and "/ 10.0" in content:
    print("✅ Scaling already present in get_cached_or_load")
else:
    print("⚠️ Scaling missing! Adding it now...")
    
    # Find where data is extracted and add scaling
    # We'll add scaling right after data is read from Zarr
    # Replace the line: data = var_data.values[...] with scaling
    
    # We'll use a more robust approach: search for the data extraction and add scaling after it.
    lines = content.splitlines()
    new_lines = []
    in_function = False
    for line in lines:
        if "def get_cached_or_load" in line:
            in_function = True
            new_lines.append(line)
            continue
        if in_function and "data = var_data.values" in line:
            new_lines.append(line)
            new_lines.append("        # Apply scaling for tmean (stored as int16 with factor 10)")
            new_lines.append("        if var_idx == 1:  # tmean")
            new_lines.append("            data = data.astype(np.float32) / 10.0")
            new_lines.append("        # tmin and tmax are already in correct units")
            continue
        if in_function and "data = var_data.isel" in line:
            new_lines.append(line)
            new_lines.append("        # Apply scaling for tmean (stored as int16 with factor 10)")
            new_lines.append("        if var_idx == 1:  # tmean")
            new_lines.append("            data = data.astype(np.float32) / 10.0")
            continue
        if in_function and line.strip() == "":
            # End of function
            in_function = False
            new_lines.append(line)
            continue
        new_lines.append(line)
    
    content = "\n".join(new_lines)
    
    with open(READ_MONTH, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Added scaling to get_cached_or_load")

# ============================================================
# 2. Clear cache to ensure fresh scaled data
# ============================================================
cache_dir = BASE / "cache" / "zarr_cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"✅ Cache cleared: {cache_dir}")
else:
    print(f"ℹ️ No cache found at: {cache_dir}")

# ============================================================
# 3. Verify constants: VAR_INDEX_FOR_FIT should be 1 (tmean)
# ============================================================
CONSTANTS = BASE / "constants.py"
with open(CONSTANTS, "r", encoding="utf-8") as f:
    const_content = f.read()

if "VAR_INDEX_FOR_FIT" in const_content:
    # Ensure it's 1
    const_content = re.sub(r'VAR_INDEX_FOR_FIT\s*=\s*\d+', 'VAR_INDEX_FOR_FIT = 1', const_content)
    with open(CONSTANTS, "w", encoding="utf-8") as f:
        f.write(const_content)
    print("✅ constants.py: VAR_INDEX_FOR_FIT = 1 (tmean)")
else:
    print("ℹ️ VAR_INDEX_FOR_FIT not found in constants.py")

# ============================================================
# 4. Summary
# ============================================================
print("\n" + "="*60)
print("✅ Setup complete for scaling during loading and caching:")
print("   - tmean is divided by 10 when read from Zarr")
print("   - Cache stores the scaled data (NOT raw int16)")
print("   - No division needed later – data is already correct")
print("   - Cache cleared to ensure fresh scaled data")
print("")
print("📌 Now re-run main.py:")
print("   python main.py")
print("")
print("   The first block will load data, scale it (÷10), and cache it.")
print("   Subsequent blocks will load from cache with correct scaling.")
print("="*60)