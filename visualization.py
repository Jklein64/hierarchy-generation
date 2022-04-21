from __future__ import annotations

from PIL import Image

import numpy as np

def show_regions(original: np.ndarray, labels: np.ndarray):
    """Visualize the label-defined regions of an 8-bit RGB(A) image by setting a regions color to the average color of pixels in the region."""
    visual = np.zeros_like(original)
    # set each non-masked region to the average color
    for label in np.ma.compressed(np.ma.unique(labels)):
        region = labels == label
        # axis=0 averages rgb(a) channels separately
        average = np.mean(original[region], axis=0).astype(int)
        visual[region] = average
    return visual


def show_constraints(original: np.ndarray, constraints: list[int, int], length=25, thickness=3):
    """Visualize the constraints on the image using black and white circles where each consecutive constraint has an additional layer of black and white.  This function requires constraints to be given as (x, y) coordinates of the image, not (row, column)!"""
    from itertools import cycle, chain, repeat
    from scipy import ndimage
    # don't overwrite the image
    visual = np.copy(original)
    height, width = np.shape(visual)[0:2]
    # create circular mask with diameter length
    y, x = np.ogrid[0:width, 0:height]
    gradient = (x - length // 2)**2 + (y - length // 2)**2
    circle =  gradient < (length // 2) ** 2
    for number, (x, y) in enumerate(constraints):
        # create array like visual but with cross
        pattern = np.zeros((height, width), dtype=bool)
        # create initial slices; x, y is flipped from row, column
        h_slice = slice(y - length // 2, y + length // 2 + length % 2)
        v_slice = slice(x - length // 2, x + length // 2 + length % 2)
        # handle out of bounds
        circle_start_h = 0
        circle_start_v = 0
        circle_stop_h = length
        circle_stop_v = length
        # out of bounds on left
        if h_slice.start < 0:
            circle_start_h = -h_slice.start
            circle_stop_h = length
        # out of bounds on right
        if h_slice.stop > width:
            circle_start_h = 0
            circle_stop_h = length // 2
        # out of bounds on top
        if v_slice.start < 0:
            circle_start_v = -v_slice.start
            circle_stop_v = length
        # out of bounds on bottom
        if v_slice.stop > height:
            circle_start_v = 0
            circle_stop_v = length // 2
        h_slice = slice(max(0, h_slice.start), min(h_slice.stop, width))
        v_slice = slice(max(0, v_slice.start), min(v_slice.stop, height))
        h_circle_slice = slice(circle_start_h, circle_stop_h)
        v_circle_slice = slice(circle_start_v, circle_stop_v)
        pattern[h_slice, v_slice] = circle[h_circle_slice, v_circle_slice]
        # two layers per number increment
        layers = number * 2 + 3
        # cycle between black and white
        color = cycle(chain(
            # four values to avoid broadcasting
            repeat([255, 255, 255, 255], thickness), 
            # and to include alpha value for black
            repeat([0, 0, 0, 255], thickness)))
        # reversed to avoid overwriting smaller regions
        for i in reversed(range(thickness - 1, layers * thickness - 1)):
            # i + 1 iterations since zero behaves differently
            visual[ndimage.binary_dilation(pattern, iterations=i + 1)] = next(color)
        # fill inside with original
        inside_pattern = ndimage.binary_dilation(pattern, iterations=thickness - 1)
        visual[inside_pattern] = original[inside_pattern]
    return visual


def show(image: np.ndarray, *, regions=None, constraints=None) -> None:
    result = image
    if regions is not None:
        result = show_regions(result, regions)
    if constraints is not None:
        result = show_constraints(result, constraints)
    Image.fromarray(result).show()