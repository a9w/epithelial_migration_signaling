"""Functions for plotting image traits."""

from .overlay_elements import overlay_random_colors
from .video import save_rgb_timelapse, save_rgb_frame

__all__ = [
    "overlay_random_colors",
    "save_rgb_timelapse",
    "save_rgb_frame",
]
