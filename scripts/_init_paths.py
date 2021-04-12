import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
lib_dir = os.path.join(this_dir, '..', 'lib')

# Add library path to PYTHONPATH
add_path(lib_dir)
