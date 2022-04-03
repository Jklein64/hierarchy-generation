from __future__ import annotations
from typing import Callable

from argparse import ArgumentParser

from skimage.segmentation import slic
from PIL import Image

import numpy as np


def main():
    parser = ArgumentParser(description="Iteratively merge superpixels of an image based on similarity and constraint locations.")
    parser.add_argument("image", help="Image file; fully transparent pixels are ignored to allow for operation on segments")
    parser.add_argument("-c", "--constraint", type=int, required=True, action="append", nargs=2, 
        # add proper labels to help text
        metavar=("x", "y"),
        # store into args.constraints
        dest="constraints",
        help="Locations of failed pixel constraints; add more by repeating this flag")

    args = parser.parse_args()

    exit()

    # TODO what if I keep all of this "opaque" stuff in terms of a labelled image?
    # TODO look into using masked arrays for the labels array

    # given original, which is a segment, possibly with opacity;
    # transparent if within bounding box but outside segment
    original = np.array(Image.open("./output/segment-1.png"))
    # labels is an array which maps each pixel location to a segment label
    labels = slic(original[..., 0:3] / 255, 200, start_label=0, multichannel=True)
    # mask labels by setting label of pixels outside segment to -1
    labels = np.where(original[..., -1] == 0 if np.shape(original)[-1] == 4 else False, -1, labels)

    # superpixels is an array of pixels for each label; iterate over range to exclude -1
    superpixels = [original[labels == label] for label in range(len(labels))]
    # find superpixel indices for superpixels which aren't empty lists due to the mask
    opaque = [label for (label, pixel) in enumerate(superpixels) if len(pixel) > 0]
    # remove superpixels which are empty lists due to the mask
    superpixels = [superpixels[label] for label in opaque]
    # create the list of pixel locations corresponding to each superpixel
    pixel_indices = [np.where(labels == label) for label in opaque]

    # create (initially) lower-triangular distances matrix
    distances = distances_matrix(superpixels, metric=wasserstein_image_distance)

    # merge connected regions
    labelled_image = connected_within_threshold(pixel_indices, np.shape(original), distances, delta=0.01)

    Image.fromarray(visualize_regions(original, labelled_image)).show()

    print(distances)


def connected_within_threshold(superpixel_pixels: list, image_shape: tuple, distances: np.ndarray, delta: float = 0.01):
    """Given a mapping from superpixel index to corresponding pixel locations, calculate the sets of connected components of a weighed undirected graph of superpixels "distances" whose weights are within a threshold delta, and return a labelled image with the given shape."""
    from scipy.sparse.csgraph import connected_components
    # labels maps index of node to a label for the group
    n, superpixel_labels = connected_components(distances < delta, directed=False)
    # create labelled image; initialize to -1 to ignore outside superpixels
    labelled = -1 * np.ones(image_shape[0:2], dtype=int)
    # set labels for each pixel for each superpixel
    for index, label in enumerate(superpixel_labels):
        labelled[superpixel_pixels[index]] = label
    return labelled


def distances_matrix(superpixels: list[np.ndarray], metric: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """Create a matrix which contains the metric-based distances between every pair of the given superpixels."""
    # create n-by-n matrix to compare distances between n superpixels
    distances = np.zeros((len(superpixels), len(superpixels)))
    # distance is symmetric, so only compare each pair once (below diagonal)
    for i, j in np.transpose(np.tril_indices(len(superpixels), k=-1)):
        distances[i, j] = metric(superpixels[i], superpixels[j])
    # fill in the rest of the distances matrix
    return distances + np.transpose(distances)


def wasserstein_image_distance(pixels_1: np.ndarray, pixels_2: np.ndarray) -> float:
    """Compute the Wasserstein or Earth Mover's distance between the given sets of integer-valued 8-bit pixels."""
    import ot
    # compute and normalize (by pixel count) color histograms for each channel
    red_1, green_1, blue_1 = map(lambda h: h / len(pixels_1), color_histograms(pixels_1))
    red_2, green_2, blue_2 = map(lambda h: h / len(pixels_2), color_histograms(pixels_2))
    # cast to float to avoid integer truncation
    pixels_1 = pixels_1[..., 0:3].astype(np.float64)
    pixels_2 = pixels_2[..., 0:3].astype(np.float64)
    # create and normalize the distance matrix
    distance = ot.dist(np.arange(0.0, 256.0)[..., np.newaxis])
    distance /= np.max(distance)
    # find optimal flows for each channel
    optimal_flow_red = ot.lp.emd(red_1, red_2, distance)
    optimal_flow_green = ot.lp.emd(green_1, green_2, distance)
    optimal_flow_blue = ot.lp.emd(blue_1, blue_2, distance)
    # derive Wasserstein distances for each channel
    wasserstein_red = np.sum(optimal_flow_red * distance)
    wasserstein_green = np.sum(optimal_flow_green * distance)
    wasserstein_blue = np.sum(optimal_flow_blue * distance)
    # sum the channel-based distances to get final metric
    return wasserstein_red + wasserstein_green + wasserstein_blue


def color_histograms(pixels: np.ndarray) -> list[np.ndarray]:
    """Maps a list of rgb pixels with integer values to three frequency charts with 256 bins each, one for each color channel.  The frequencies are NOT normalized."""
    r, g, b = np.transpose(pixels)[0:3]
    histograms = []
    for channel in (r, g, b):
        # need explicit integer type since np.float64 is default
        # needs to be big enough to store a count! uint8 won't do!
        histogram = np.zeros((2 ** 8), dtype=np.uint64)
        values, counts = np.unique(channel, return_counts=True)
        histogram[values] = counts
        histograms.append(histogram)
    return histograms


def visualize_regions(original: np.ndarray, labelled: np.ndarray):
    """Visualize the label-defined regions of an 8-bit RGB(A) image by setting a regions color to the average color of pixels in the region."""
    visual = np.zeros_like(original)
    # get non-negative labels
    labels = np.unique(labelled)[np.unique(labelled) >= 0]
    # set each region to the average color
    for label in labels:
        region = labelled == label
        # axis=0 averages rgb(a) channels separately
        average = np.mean(original[region], axis=0).astype(int)
        visual[region] = average
    return visual


if __name__ == "__main__":
    main()