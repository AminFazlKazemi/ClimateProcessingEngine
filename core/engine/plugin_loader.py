#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/engine/plugin_loader.py
Auto-load plugins from directory
"""

import os
import importlib
import inspect
from core.engine.distribution_plugin import DistributionPlugin

def load_plugins(plugin_dir="plugins/distributions"):
    plugins = {}
    for filename in os.listdir(plugin_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            module = importlib.import_module(f"{plugin_dir.replace('/', '.')}.{module_name}")
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, DistributionPlugin) and obj != DistributionPlugin:
                    instance = obj()
                    plugins[instance.code] = instance
    return plugins
