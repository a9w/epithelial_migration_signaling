"""
Segment cells in each frame of a timelapse, then track them.
Segments registered timelapses by default if present. 

Input: A tif timelapse of an egg chamber with cell edges labeled.

Output: A tif timelapse with labeled cells.
"""

# Import packages
import os
import sys
import numpy as np
from imageio import volread

# Add the tools directory to Python's path list and import imtools
from functions.track import TrackedTimelapse
from functions.utils import select_files

# Hard code the path to the timelapses
DATA_DIR = ("../data/membrane_protrusivity_polarity/sample_data/")

# Get paths to intensities timelapses
datasets = select_files(DATA_DIR, [".tif"])
for dataset in datasets:
    basename = dataset["basename"]
    print(f"Segmenting and tracking {basename}")

    # Import the raw images and convert to an array
    try:
        ims_registered_path = os.path.join(DATA_DIR, f'{basename}_reg.tif')
        ims = volread(ims_registered_path)
    except FileNotFoundError:
        ims = volread(dataset[".tif"])

    # Make a TrackedTimelapse object
    tt = TrackedTimelapse(ims, basename=basename, out_dir=DATA_DIR)
