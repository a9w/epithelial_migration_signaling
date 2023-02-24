"""
Find the edge and non-edge basal surface regions of an epithelium
and measure the mean fluorescence intensity within each region.

Input a folder with intensities images or image stack tifs
(with 1 channel to be measured), segmented cell image tif.

Outputs a CSV with rows for each sample and cols for their condition, number,
and mean intensities across the tissue, at edges, at interiors, and
at edges-interiors.
"""

# Import packages
import numpy as np
import pandas as pd
from imageio import imread, volread

# Internal functions
from functions.segment import (cell_edges_mask,
                                cell_interiors_mask,
                                select_in_field)
from functions.utils import select_files

# Set locations of images (intensities and segmented), output
DATA_DIR =('../data/Sra1GFP_level_polarity/')
OUT_DIR =('../data/Sra1GFP_level_polarity/')

# Set total channels in intensities image file,
# index of channel to measure,
# name of measured channel (for output naming)
CHANNELS_TOTAL = 3
INTENSITIES_CHANNEL_IND = 1
INTENSITIES_CHANNEL_NAME = "Sra1GFP"

# Set how wide the edge regions are
EDGE_DIL_FACTOR = 5

# Get paths to intensities and segmented images
file_info = select_files(DATA_DIR, ['.tif','_seg_corr.tif'])

# Import images, make masks, measure mean intensities
condition = []
sample_nums = []
mean_tissue_ls = []
mean_edges_ls = []
mean_interiors_ls = []
mean_edges_minus_interiors_ls = []

for i in range(len(file_info)):
    im_intensities_path = file_info[i]['.tif']
    im_lab_path = file_info[i]['_seg_corr.tif']

    if CHANNELS_TOTAL > 1:
        im_intensities_raw = volread(im_intensities_path)
        im_intensities = im_intensities_raw[INTENSITIES_CHANNEL_IND]
    else:
        im_intensities = imread(im_intensities_path)
    im_lab = imread(im_lab_path)

    # Get condition and sample number
    basename = file_info[i]['basename']
    sample_num = basename.split('_')[-1]
    sample_nums.append(sample_num)
    condition.append(basename.split('_' + sample_num)[0])

    # Track progress
    print(f'Analyzing image {str(i)} out of {str(len(file_info))}, {basename}')

    # Make the masks
    tissue_mask = im_lab > 0
    tissue_mask = select_in_field(im_lab, tissue_mask)
    edges_mask = cell_edges_mask(im_lab, EDGE_DIL_FACTOR)
    interiors_mask = cell_interiors_mask(im_lab, EDGE_DIL_FACTOR)

    # Measure mean intensity within masks, add to lists
    mean_tissue_ls.append(np.mean(im_intensities[tissue_mask]))
    mean_edges = np.mean(im_intensities[edges_mask])
    mean_interiors = np.mean(im_intensities[interiors_mask])
    mean_edges_ls.append(mean_edges)
    mean_interiors_ls.append(mean_interiors)
    mean_edges_minus_interiors_ls.append(mean_edges - mean_interiors)

# Construct df of sample info and intensities, output as csv
col_names = ['condition', 'sample_num', 'mean_tissue', 'mean_edges',
             'mean_interiors', 'mean_edges_minus_interiors']
df = pd.DataFrame(list(zip(condition, sample_nums, mean_tissue_ls,
                           mean_edges_ls, mean_interiors_ls,
                           mean_edges_minus_interiors_ls)),
                 columns = col_names)
df_sorted = df.sort_values(['condition', 'sample_num'])
df_sorted.reset_index(inplace=True, drop=True)
df_path = (OUT_DIR + INTENSITIES_CHANNEL_NAME +
            '_mean_intensity_edge_interior_sample.csv')
df_sorted.to_csv(path_or_buf = df_path)
