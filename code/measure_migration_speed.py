"""
Measure tissue migration rate from tracked cells.

Finds the net displacement vector of all near-medial cells
in all frames. Migration speed is the length of this vector.

Also outputs the medial mask and medial cells used for measurement.
"""

# Import packages
import os
import numpy as np
from math import degrees
from imageio import imwrite, volread, volwrite
import pandas as pd
from skimage.measure import regionprops

from functions.utils import select_files, cart_to_pol
from functions.segment import select_in_field, tissue_axis_mask
from functions.measure import tissue_medial_orientation

# Data location and keys for tracked cells and tissue mask files
DATA_DIR = ("../data/membrane_protrusivity_polarity/sample_data/")
OUT_DIR = ("../data/membrane_protrusivity_polarity/")

# Set timelapse scale parameters, width of region to include as medial
FRAMES_PER_MIN = 2
MIN_TO_ANALAYZE = 20

# Membrane protrusion dataset settings (Fig. 5B, gray background)
PIXELS_PER_UM = 11.30
MEDIAL_REGION_WIDTH = 250 # px
KEYS = ["_tissue_mask.tif", "_tracked_refined.tif"]

# Get dataset basenames, paths
datasets = select_files(DATA_DIR, KEYS)

if len(datasets) == 0:
    print("No datasets to measure")

conditions = []
sample_nums = []
basenames = []
speeds = []
dirs_rad = []
dirs_deg = []

for dataset in datasets:
    basename = dataset["basename"]
    print(f"Measuring migration speed of {basename}")
    ims_tracked = volread(dataset[KEYS[1]])
    tissue_masks_int = volread(dataset[KEYS[0]])
    tissue_masks_bool = tissue_masks_int.astype(bool)
    medial_ori = tissue_medial_orientation(tissue_masks_bool[0])
    props = regionprops(tissue_masks_int[0])[0]
    centroid = props.centroid # (r,c)

    # Check for medial mask and medial cells tifs, else make them
    medial_mask_path = os.path.join(DATA_DIR, f"{basename}_medial_mask.tif")
    medial_cells_path = os.path.join(DATA_DIR, f"{basename}_tracked_refined_medial.tif")

    try:
        medial_mask = volread(medial_mask_path)
        ims_medial = volread(medial_cells_path)
    except FileNotFoundError:
        # Find medial cells. Uses frame 0 centroid, orientation
        # to make medial mask.
        medial_mask = tissue_axis_mask(
            tissue_masks_bool[0], MEDIAL_REGION_WIDTH, medial_ori, centroid
        )
        ims_medial = np.zeros_like(ims_tracked)
        for t, im_tracked in enumerate(ims_tracked):
            ims_medial[t] = im_tracked * select_in_field(im_tracked,
                                                            medial_mask)

        # Output medial mask, medial cells (quality control)
        imwrite(medial_mask_path, medial_mask.astype('uint8'))
        volwrite(medial_cells_path, ims_medial)

    # Find medial cell centroids
    frames_to_analyze = MIN_TO_ANALAYZE * FRAMES_PER_MIN + 1
    df_centroids_ls = []
    for t, im_medial in enumerate(ims_medial[:frames_to_analyze]):
        labels = []
        centroid_rows = []
        centroid_cols = []
        for region in regionprops(im_medial):
            label = region.label
            labels.append(label)
            centroid_row, centroid_col = region.centroid
            centroid_rows.append(centroid_row)
            centroid_cols.append(centroid_col)

        df_frame = pd.DataFrame({"cell":labels,
                                 "frame":t,
                           "centroid_row":centroid_rows,
                           "centroid_col":centroid_cols})
        df_centroids_ls.append(df_frame)

    # Find displacements for each cell between consecutive frames
    df_disps_list = []
    for t in range(1,len(df_centroids_ls)):
        # Find cells present in frames t-1 and t
        cells_t0 = list(df_centroids_ls[t-1]['cell'])
        cells_t1 = list(df_centroids_ls[t]['cell'])
        cells_t0_set = set(cells_t0)
        shared_cells_set = cells_t0_set.intersection(cells_t1)
        shared_cells_list = list(shared_cells_set)

        r_disps = []
        c_disps = []
        dir_disps = []
        mag_disps = []
        for i, cell in enumerate(shared_cells_list):
            df_t0 = df_centroids_ls[t-1]
            df_t1 = df_centroids_ls[t]
            r_t0 = float(df_t0[df_t0['cell']==cell]['centroid_row'])
            c_t0 = float(df_t0[df_t0['cell']==cell]['centroid_col'])
            r_t1 = float(df_t1[df_t1['cell']==cell]['centroid_row'])
            c_t1 = float(df_t1[df_t1['cell']==cell]['centroid_col'])
            r_disp = (r_t1 - r_t0) / PIXELS_PER_UM * FRAMES_PER_MIN
            c_disp = (c_t1 - c_t0) / PIXELS_PER_UM * FRAMES_PER_MIN
            dir_disp, mag_disp = cart_to_pol(c_disp, r_disp)
            r_disps.append(r_disp)
            c_disps.append(c_disp)
            dir_disps.append(dir_disp)
            mag_disps.append(mag_disp)

        df_disps_frame = pd.DataFrame({"cell":shared_cells_list,
                                        "r_disp":r_disps,
                                        "c_disp":c_disps,
                                        "dir_disp":dir_disps,
                                        "mag_disp":mag_disps,
                                        "frame_start":t-1,
                                        "frame_end":t})
        df_disps_list.append(df_disps_frame)

    df_disps = pd.concat(df_disps_list)
    r_disp_mean = np.mean(df_disps['r_disp'])
    c_disp_mean = np.mean(df_disps['c_disp'])
    dir_mean_rad, speed_mean = cart_to_pol(c_disp_mean, r_disp_mean)
    dir_mean_deg = degrees(dir_mean_rad)

    basename = dataset["basename"]
    sample_num = basename.split("_")[-1]
    condition = basename.split(f"_{sample_num}")[0]
    conditions.append(condition)
    sample_nums.append(sample_num)
    basenames.append(basename)
    speeds.append(speed_mean)
    dirs_rad.append(dir_mean_rad)
    dirs_deg.append(dir_mean_deg)

# Make dataframe, output as CSV
df_migration = pd.DataFrame({"condition":conditions,
                            "sample_num":sample_nums,
                            "sample":basenames,
                            "mig_speed_um_per_min":speeds,
                            "mig_dir_rad":dirs_rad,
                            "mig_dir_deg":dirs_deg})
out_path = os.path.join(OUT_DIR, "migration_speed_sample.csv")
df_migration = df_migration.sort_values(['sample'])
df_migration.reset_index(inplace=True, drop=True)
df_migration.to_csv(path_or_buf = out_path)
