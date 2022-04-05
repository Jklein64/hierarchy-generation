from __future__ import annotations
from typing import Callable

from argparse import ArgumentParser

from scipy.sparse.csgraph import connected_components
from skimage.segmentation import slic
from PIL import Image

import ot, numpy as np


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

    # TODO visualize constraints on the image
    # TODO how to show constraint order????
    # 65 mins using masked arrays!

    # original is an 8-bit rgb(a) image, possibly with opacity;
    # transparent if within bounding box but outside segment
    original = np.array(Image.open(args.image))

    labels = np.ma.array(
        # labels maps pixel location to a superpixel label
        slic(original[..., 0:3] / 255, 200, start_label=0),
        # don't mask anything if no alpha channel, otherwise mask transparent pixels
        mask=np.shape(original)[-1] == 4 and original[..., -1] == 0)

    distances = distances_matrix(original, labels, metric=wasserstein_image_distance)
    labelled_image = connected_within_threshold(labels, distances, delta=0.01)

    Image.fromarray(visualize_regions(original, labelled_image)).show()

    print(distances)


def connected_within_threshold(superpixels: np.ndarray, distances: np.ndarray, delta: float = 0.01):
    """Given a mapping from pixel location to superpixel label as well as a weighted adjacency matrix, calculate the sets of connected components of a weighed undirected graph of superpixels "distances" whose weights are within a threshold delta, and return a newly labelled image."""
    # labels maps index of node to a label for the group
    n, superpixel_labels = connected_components(distances < delta, directed=False)
    # create labelled image; keep superpixels' mask
    # note that zeros_like() doesn't exist for np.ma
    labels = np.ma.copy(superpixels)
    # set labels for each pixel for each superpixel
    for index, label in enumerate(superpixel_labels):
        labels[superpixels == index] = label
    return labels


def distances_matrix(original: np.ndarray, superpixels: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """Create a matrix with the metric-based distances between every pair of the given superpixels implied by the original and labelled images."""
    # store list of valid superpixel labels
    unique_labels = np.ma.compressed(np.ma.unique(superpixels))
    # pixels is a list of pixel values for the n'th superpixel
    pixels = [original[superpixels == label] for label in unique_labels]
    # create n-by-n matrix to compare distances between n superpixels
    distances = np.zeros((len(unique_labels), len(unique_labels)))
    # distance is symmetric, so only compare each pair once (below diagonal)
    for i, j in np.transpose(np.tril_indices(len(unique_labels), k=-1)):
        distances[i, j] = metric(pixels[i], pixels[j])
    # fill in the rest of the distances matrix
    return distances + np.transpose(distances)


def wasserstein_image_distance(pixels_1: np.ndarray, pixels_2: np.ndarray) -> float:
    """Compute the Wasserstein or Earth Mover's distance between the given sets of integer-valued 8-bit pixels."""
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


def visualize_regions(original: np.ndarray, labels: np.ndarray):
    """Visualize the label-defined regions of an 8-bit RGB(A) image by setting a regions color to the average color of pixels in the region."""
    visual = np.zeros_like(original)
    # set each non-masked region to the average color
    for label in np.ma.compressed(np.ma.unique(labels)):
        region = labels == label
        # axis=0 averages rgb(a) channels separately
        average = np.mean(original[region], axis=0).astype(int)
        visual[region] = average
    return visual


if __name__ == "__main__":
    main()