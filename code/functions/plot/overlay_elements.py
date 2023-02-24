"""
Functions overlaying visible elements on an image.

These can be stacked up to produce more complex figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from ..segment import select_in_field
from ..utils import validate_mask


def overlay_random_colors(
    im, mask=None, periphery_excluded=True, alpha=0.4, same_seed=True
):
    """
    Generate an RGBA image of regions.

    Parameters
    ----------
    im : 2D ndarray
        Regions labeled with unique values
    mask : bool ndarray
        Optional mask, same shape as im
    periphery_excluded : bool
        Whether regions touching the border or mask
        should be included in the plot
    alpha : float
        Transparency from 0 to 1
    same_seed : bool
        If True, im_rgba colors will be consistent across repeated function
        calls for one input im

    Returns
    -------
    im_rgba : ndarray with dimensions (y,x,4)
        Regions labeled with random
    """
    mask = validate_mask(im, mask)

    # If periphery_excluded=True, set mask- and border-adjacent regions to zero
    if periphery_excluded:
        mask = mask * select_in_field(im, mask)

    # Make an array of unique cell labels
    im_relabeled = label(im * mask, background=-1) - 1
    unique_labels = np.unique(im_relabeled)

    # Make a set of random colors
    color_list = []

    for label_new in unique_labels:
        if same_seed == True:
            # find label in original image
            label_orig = np.unique(im * (im_relabeled == label_new))[-1]
            # Use the label as random seed
            np.random.seed(label_orig)
        elif same_seed == False:
            # No fixed seed
            pass
        color_list.append([np.random.rand() for i in range(3)] + [alpha])
    color_array = np.array(color_list)

    # Make an RGBA image
    rows, cols = np.shape(im_relabeled)
    im_rgba = np.zeros((rows, cols, 4))

    # Plot the colors on the original image
    im_rgba = color_array[im_relabeled]

    # Set masked regions to transparent
    im_rgba[:, :, 3][mask == False] = 0

    return im_rgba
