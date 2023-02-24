"""Segment hemijunctions from a directory of datasets."""

# Import packages
import os
import sys
from imageio import volread, volwrite
import numpy as np

# Add the tools directory to Python's path list and import imtools
from functions.segment import (
    segment_hemijunctions_timelapse,
    largest_object_mask_timelapse,
)
from functions.utils import select_files

# Set data directory, tracked cells identifier
DATA_DIR = ("../data/membrane_protrusivity_polarity/sample_data/")
KEYS = [".tif","_tracked_corr.tif"]

# Get file info
datasets = select_files(DATA_DIR, KEYS)

# Iterate over datasets in the DATA_DIR, then segments hemijunctions of each one
for dataset in datasets:
    basename = dataset["basename"]
    print(f"***** {basename} *****")
    hjs_path = os.path.join(DATA_DIR, f"{basename}_tracked_hjs.tif")
    tracked_refined_path = os.path.join(DATA_DIR, f"{basename}_tracked_refined.tif")
    ims_mask_path = os.path.join(DATA_DIR, f"{basename}_tissue_mask.tif")

    # Check for existing hemijunctions and refined cells files, or make them
    try:
        ims_labels_tracked_refined = volread(tracked_refined_path)
        ims_labels_tracked_hjs = volread(hjs_path)
        print("Found existing segmented hemijunctions")
    except FileNotFoundError:
        print("Loading intensities and labeled cell datasets")
        ims_intensities = volread(dataset[".tif"])
        ims_labels = volread(dataset["_tracked_corr.tif"])

        # Check for existing tissue mask file or make one
        try:
            ims_mask = volread(ims_mask_path)
        except FileNotFoundError:
            print("Segmenting the tissue")
            ims_mask = largest_object_mask_timelapse(ims_intensities)
            volwrite(ims_mask_path, ims_mask)

        print("Segmenting hemijunctions")
        (
            ims_labels_tracked_refined,
            ims_labels_tracked_hjs,
        ) = segment_hemijunctions_timelapse(ims_labels, ims_intensities)

        print("Saving volumes")
        refined_path = os.path.join(DATA_DIR, f"{basename}_tracked_refined.tif")
        volwrite(refined_path, ims_labels_tracked_refined)
        hjs_path = os.path.join(DATA_DIR, f"{basename}_tracked_hjs.tif")
        volwrite(hjs_path, ims_labels_tracked_hjs)
