import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)