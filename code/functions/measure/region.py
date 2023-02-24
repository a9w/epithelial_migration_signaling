"""Functions to compute traits at the single-region level."""

import numpy as np
import skfmm
from skimage.measure import find_contours, approximate_polygon
from ..segment import interface_endpoints_coords, edge_between_neighbors
from ..utils import points_to_angle


def measure_one_hemijunction(cell_s, cell_r, interface):
    """
    Measure traits of a single hemijunction.

    Parameters
    ----------
    cell_s : 2D bool ndarray
        Pixels in the sending cell are True, rest are False
    cell_r : 2D bool ndarray
        Pixels in the receiving cell are True, rest are False
    interface : 2D bool ndarray
        Pixels in the interface are True, rest are False

    Returns
    -------
    hj_traits : dict with these keys
        "hj_area_px2" (int) number of pixels in the hemijunction
        "tip_coords" (tuple of ints) coordinates of hemijunction tip
        "base_coords" (tuple of ints) coordinates of hemijunction base
        "prot_len_px" (float) protrusion length in px units
        "prot_angle_rad" (float) angle from base to tip in radians
        "prot_angle_deg" (float) angle from base to tip in degrees
        "edge_len_nonstrt_px" (float) the curving, pixelated length of the interface
        "edge_len_strt_px" (float) the straight-line length of the interface
        "edge_angle_rad" (float)
        "edge_angle_deg" (float)
        "endpoint_1_coords" (tuple of floats) coordinates of one interface endpoint
        "endpoint_2_coords" (tuple of floats) coordinates of other interface endpoint
    """
    # Measure hemijunction traits
    hj_area_px2 = np.sum(np.logical_and(cell_r, interface))
    hj_traits = {"hj_area_px2": hj_area_px2}
    tip, base, length_internal = protrusion_length_internal_path(
        cell_s, cell_r, interface
    )
    hj_traits["tip_coords"] = tip
    hj_traits["base_coords"] = base
    hj_traits["prot_len_px"] = length_internal
    angle_rad = points_to_angle(base, tip)
    hj_traits["prot_angle_rad"] = angle_rad
    hj_traits["prot_angle_deg"] = np.degrees(angle_rad)
    # Measure "edge" traits (traits of the place where two cells meet)
    hj_traits["edge_len_nonstrt_px"] = interface_length_wiggly(cell_s, cell_r)
    hj_traits["edge_len_strt_px"] = interface_length_segment(cell_s, cell_r)
    e1, e2 = interface_endpoints_coords(cell_s, cell_r)
    edge_rad = points_to_angle(e1, e2)
    hj_traits["edge_angle_rad"] = edge_rad
    hj_traits["edge_angle_deg"] = np.degrees(edge_rad)
    hj_traits["endpoint_1_coords"] = e1
    hj_traits["endpoint_2_coords"] = e2
    return hj_traits


def interface_length_segment(cell_a, cell_b):
    """
    Measure straight-line length of an interface.

    Parameters
    ----------
    cell_a, cell_b: 2D bool ndarrays
        Pixels in the cells are True, rest are False

    Returns
    -------
    length : float
        Segment length connecting the interface corners
    """
    e1, e2 = interface_endpoints_coords(cell_a, cell_b)
    length = np.linalg.norm(np.array(e1) - np.array(e2))
    return length


def interface_length_wiggly(cell_a, cell_b):
    """
    Measures the curvy length of an interface.

    Parameters
    ----------
    call_a, cell_B : 2D bool ndarrays
        Pixels in the cells are True, rest are False

    Returns
    -------
    length : float
        Wiggly length of cell interface, calculated as the perimeter of the
        interface mask, divided by 2, minus 2 (to account for its width)
    """
    edge = edge_between_neighbors(cell_a, cell_b)
    length = polygonal_perimeter(edge) / 2 - 2
    return length


def polygonal_perimeter(shape, tolerance=1):
    """
    Use the polygonal approximation of a pixelated shape to estimate its perimeter.

    Parameters
    ----------
    shape: 2D bool ndarray
        Pixels in the shape are True, rest are False.
    tolerance: float
        "Maximum distance from original points of polygon to approximated polygonal
        chain. If tolerance is 0, the original coordinate array is returned".
        Higher tolerance means fewer vertices in the polygon approximation.

    Returns
    -------
    total : float
        Calculated as the lengths of a series of line segments of all contours.
    """
    contours = find_contours(shape, 0.5, fully_connected="high")
    total = 0
    for contour in contours:
        coords = approximate_polygon(contour, tolerance=tolerance)
        # Distance from last coordinate to first
        perimeter = np.linalg.norm(coords[-1] - coords[0])
        # Add the distances between the rest of the successive coordinate pairs
        for i in range(1, coords.shape[0]):
            segment_length = np.linalg.norm(coords[i - 1] - coords[i])
            perimeter += segment_length
        total += perimeter
    return total


def protrusion_length_internal_path(cell_s, cell_r, interface):
    """
    Measure length of a protrusion as an internal path.

    Each protrusion is defined in terms of a sending cell,
    a receiving cell, and an interface mask that falls between
    the two cells. Here, the FMM algorithm is used to trace the path
    from the sending cell into the hemijunction. The pixel with the
    highest value is called the 'tip'. The length of the path to that
    point is called the 'length'.

    Then again FMM is used to trace from the tip back to the sending cell.
    The pixel with the lowest value is called the 'base'.

    Parameters
    ----------
    cell_s : 2D bool ndarray
        Pixels in the sending cell are True, rest are False
    cell_r : 2D bool ndarray
        Pixels in the receiving cell are True, rest are False
    interface : 2D bool ndarray
        Pixels in the interface are True, rest are False

    Returns
    -------
    length : float
        Internal path length from cell_s to farthest point in interface
    tip_single : tuple of 2 ints
        Coordinates of the farthest point in interface, and if there are
        multiple tied pixels, take the first of them
    base_single : tuple of 2 ints
        Coordinates of point on sending cell that is closest
        to the protrusion tip, and if there are multiple tied pixels, take
        the first of them
    """
    # Make some boolean masks that will be needed
    not_r_nor_s = np.invert(np.logical_or(cell_r, cell_s))
    cell_r_not_interface = np.logical_and(cell_r, np.invert(interface))

    # FMM from cell_s into interface
    mask = np.logical_or(cell_r_not_interface, not_r_nor_s)
    phi = np.ma.MaskedArray(np.invert(cell_s), mask)
    dist_in_interface_from_s = skfmm.distance(phi)

    # Maximum value in interface is the "length". Note that is possible
    # for multiple pixels to 'tie' for farthest; take first by default.
    length = np.max(dist_in_interface_from_s)
    tip = np.nonzero(dist_in_interface_from_s == length)

    # March from tip to cell_s
    tip_zero = np.ones(np.shape(cell_s))
    tip_zero[tip[0][0], tip[1][0]] = 0
    phi = np.ma.MaskedArray(tip_zero, mask)
    dist_in_cell_and_int_from_tip = skfmm.distance(phi)

    # Get coordinates of base
    dist_in_cell = np.ma.MaskedArray(dist_in_cell_and_int_from_tip, cell_r)
    base = np.nonzero(dist_in_cell == np.min(dist_in_cell))

    # If there are multiple tip and base points, just keep first ones
    tip_out = (tip[0][0], tip[1][0])
    base_out = (base[0][0], base[1][0])

    return tip_out, base_out, length


def protrusion_angle(tip, base):
    """
    Return angle from base to tip.

    Parameters
    ----------
    tip : tuple of 2 ints
    base: tuple of 2 ints

    Returns
    -------
    angle : float
        Angle from base to tip, in radians
    """
    tip_coords = np.ravel(np.array(tip))
    base_coords = np.ravel(np.array(base))
    protrusion_vector = tip_coords - base_coords
    angle = np.arctan2(protrusion_vector[0], protrusion_vector[1])
    return angle
