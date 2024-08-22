import sys
import os

def setup_base_dir(levels=2):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(levels):
        base_dir = os.path.dirname(base_dir)
    sys.path.insert(0, base_dir)
    return base_dir