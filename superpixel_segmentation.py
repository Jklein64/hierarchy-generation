from __future__ import annotations
from skimage.segmentation import slic
from PIL import Image

import numpy as np


def main():
    # given original, which is a segment, possibly with opacity;
    # transparent if within bounding box but outside segment
    original = np.array(Image.open("./output/segment-1.png"))
    # slic() expects images to be floating point and 3-channel
    image = original[..., 0:3] / 255
    # labels is an array which maps each pixel location to a segment label
    labels = slic(image, 200, start_label=0, multichannel=True)

    # superpixels is an array of pixels for each label
    superpixels = [original[labels == label] for label in np.unique(labels)]
    # isolate superpixels which overlap the segment region of interest
    opaque = [label for (label, p) in enumerate(superpixels) if np.any(p[..., -1] != 0)]
    superpixels = [superpixels[label] for label in opaque]
    # create the list of pixel locations corresponding to each visible superpixel
    pixel_indices = [np.where(labels == label) for label in opaque]

    # create (initially) lower-triangular distances matrix
    distances = np.zeros((len(superpixels), len(superpixels)))
    # we are interested in values below the diagonal but not including it, so k = -1
    # the diagonal would measure the distance from each superpixel to itself
    for i, j in np.column_stack(np.tril_indices(len(superpixels), k=-1)):
        distances[i, j] = wasserstein_image_distance(superpixels[i], superpixels[j])
    # make distances an actual adjacency matrix
    distances = distances + np.transpose(distances)

    # merge connected regions
    regions = []
    for component in connected_within_threshold(distances, 0.02):
        # map superpixel index to the superpixel's pixel locations
        indices = [pixel_indices[i] for i in component]
        # concatenate; must be manual since np.where is a tuple
        concatenated_rows = []
        concatenated_cols = []
        for rows, cols in indices:
            concatenated_rows.append(rows)
            concatenated_cols.append(cols)
        # tuple allows for slicing of numpy arrays
        regions.append((
            np.concatenate(concatenated_rows),
            np.concatenate(concatenated_cols)))
    for region in regions:
        image = np.zeros_like(original)
        image[region] = original[region]

    
# superpixels_to_join = np.column_stack(distance_indices)[np.where(distances[distance_indices] < 0.01)]
    print(distances)

def connected_within_threshold(weighed: np.ndarray, delta: float = 0.01):
    """Calculate the sets of connected components of a weighed undirected graph whose weights are within a threshold delta."""
    from scipy.sparse.csgraph import connected_components
    # labels maps index of node to a label for the group
    n, labels = connected_components(weighed < delta, directed=False)
    # need the first element of np.nonzero since it wraps itself in a tuple
    return [np.nonzero(labels == label)[0] for label in range(n)]



def wasserstein_image_distance(pixels_1: np.ndarray, pixels_2: np.ndarray) -> float:
    """Compute the Wasserstein or Earth Mover's distance between the given sets of integer-valued 8-bit pixels."""
    import ot
    # ignore pixels from the superpixels which are outside of the segment,
    # but where the superpixel still has some pixels inside the segment
    pixels_1 = pixels_1[pixels_1[..., -1] != 0]
    pixels_2 = pixels_2[pixels_2[..., -1] != 0]
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


if __name__ == "__main__":
    main()