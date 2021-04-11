import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(this_dir)

# Add project path to PYTHONPATH
add_path(parent_dir)
