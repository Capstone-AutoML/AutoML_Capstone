"""
Initialize the prelabelling module and set up import paths
"""
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

# Import commonly used functions from utils
from utils import detect_device
