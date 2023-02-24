"""Segment tissues for a directory of timelapse datasets"""

# Import packages
import os
import sys
from imageio import volread, volwrite

# Add the tools directory to Python's path list and import imtools
from functions.segment import largest_object_mask_timelapse
from functions.utils import select_files

# Data location and string to identify the intensities timelapses
DATA_DIR = ("../data/membrane_protrusivity_polarity/sample_data/")
KEY = ".tif"

# Get dataset basenames, paths
datasets = select_files(DATA_DIR, KEY)

# Make tissue masks for each intensities timelapse, if not already present
for dataset in datasets:
    basename = dataset["basename"]
    print(f"***** {basename} *****")
    out_path =  os.path.join(DATA_DIR, f"{basename}_tissue_mask.tif")
    # Check for existing tissue_mask, otherwise make one
    try:
        tissue_mask = volread(out_path)
        print("Found existing tissue mask.")
    except FileNotFoundError:
        print("Making tissue mask.")
        ims = volread(dataset[KEY])
        tissue_mask = largest_object_mask_timelapse(ims)
        tissue_mask = tissue_mask.astype('uint8')
        volwrite(out_path, tissue_mask)
