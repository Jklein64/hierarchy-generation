from __future__ import annotations
from typing import Callable

from argparse import ArgumentParser

from scipy.sparse.csgraph import connected_components
from scipy.ndimage import binary_dilation
from skimage.segmentation import slic
from PIL import Image

import numpy as np

from metrics import wasserstein_image_distance, average_color_distance
from visualization import show


def main():
    parser = ArgumentParser(description="Iteratively merge superpixels of an image based on similarity and constraint locations.")
    parser.add_argument("image", help="Image file; fully transparent pixels are ignored to allow for operation on segments")
    parser.add_argument("-c", "--constraint", type=int, required=True, action="append", nargs=2, 
        # add proper labels to help text
        metavar=("row", "column"),
        # store into args.constraints
        dest="constraints",
        help="Locations of failed pixel constraints; add more by repeating this flag")

    args = parser.parse_args()

    # turn (x, y) array into (row, col) tuple to allow for indexing
    # note that constraints is sorted with most recent as last
    constraints = [tuple(c) for c in args.constraints]
    # original is an 8-bit rgb(a) image, possibly with opacity;
    # transparent if within bounding box but outside segment
    original = np.array(Image.open(args.image))

    # show the original image
    show(original, constraints=constraints)

    labels = np.ma.array(
        # labels maps pixel location to a superpixel label
        slic(original[..., 0:3] / 255, 200, start_label=0),
        # don't mask anything if no alpha channel, otherwise mask transparent pixels
        mask=np.shape(original)[-1] == 4 and original[..., -1] == 0)

    # show the initial superpixel segmentation
    show(original, regions=labels, constraints=constraints)

    # merge neighbors within threshold to reduce the total number of superpixels
    neighbors = neighbor_matrix(original, labels, metric=wasserstein_image_distance)
    # invert neighbors to represent distance instead of similarity; delta is arbitrary
    # FIXME justify delta choice
    merged_rag = connected_within_threshold(labels, 1 - neighbors, delta=0.01)

    # show the image after merging the region adjacency graph
    show(original, regions=merged_rag, constraints=constraints)

    distances = distances_matrix(original, merged_rag, metric=average_color_distance)

    merged = constrained_division(merged_rag, distances, constraints[0], constraints[1])

    # show the image after applying the first two constraints
    show(original, regions=merged, constraints=constraints)


    # TODO fix the names
    # TODO generalize to n constraints
    label_containing_3 = merged[constraints[2]]
    # ignore the last constraint
    for c in constraints[0:-1]:
        pixel_has_3 = merged == label_containing_3
        # pixel at constraint is part of region of interest
        if pixel_has_3[c]:
            break
    masked_merged_rag = np.ma.array(merged_rag, mask=(merged != label_containing_3))
    # TODO AAAAAAA need to mask out more superpixels from the distances

    # which superpixels are remaining? make a list of labels
    removed_superpixels = np.unique(merged_rag[merged != label_containing_3])
    removed_mask = np.ones_like(distances).astype(bool)
    for i in removed_superpixels:
        # remove i'th row and column
        removed_mask[i, ...] = False
        removed_mask[..., i] = False
    d = np.shape(distances)[0] - len(removed_superpixels)
    # remove each of those from the distances matrix
    # m
    m = constrained_division(masked_merged_rag, np.reshape(distances[removed_mask], (d, d)), c, constraints[2])
    pass

    # TODO do a constrained division iterating through the constraints.
    # Given time-consecutive constraints 1, 2, and 3, divide between 1 and 2
    # first.  Then, mask out the region that doesn't include 3 and divide
    # again between 3 and the label for the unmasked region.


def constrained_division(superpixels: np.ndarray, distances: np.ndarray, old_constraint: tuple[int, int], new_constraint: tuple[int, int]):
    """Given a possibly-transparent image's masked superpixel segmentation, the pairwise distance between those superpixels (after RAG merging), and two constraints, divide the image into two semantic regions such that each constraint is in its own region.  The labels returned from this method correspond to the given constraints."""
    # binary search to find the largest value of 
    # delta which still separates the constraints
    low = 0
    high = np.max(distances)
    delta = (high + low) / 2
    # FIXME justify threshold choice
    threshold = high * 0.001
    # create initial merged image
    merged = connected_within_threshold(superpixels, distances, delta)
    # find which superpixel each constraints belongs to
    a = merged[old_constraint]
    b = merged[new_constraint]
    # a == b so that it always separates the constraints
    while a == b or (high - low) > threshold:
        if a == b:
            # too general
            high = delta
        else:
            # specific enough
            low = delta
        # reset loop variables
        delta = (high + low) / 2
        merged = connected_within_threshold(superpixels, distances, delta)
        a = merged[old_constraint]
        b = merged[new_constraint]
    # assign regions not containing either constraint to the older one
    for label in np.unique(merged[~np.ma.getmask(merged)]):
        # merge all but the region containing the most recent constraint
        merged[merged == label] = b if label == b else a
    return merged
    
    


def neighbor_matrix(original: np.ndarray, superpixels: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """Create a weighed adjacency matrix between every pair of superpixels which neighbors each other.  Weighed region adjacency graph!  The resulting matrix is normalized so that the weights are within [0, 1], and then rescaled so that higher values correspond to higher similarity."""
    # store list of valid superpixel labels
    unique_labels = np.ma.compressed(np.ma.unique(superpixels))
    # pixels is a list of pixel values for the n'th superpixel
    pixels = [original[superpixels == label] for label in unique_labels]
    # create n-by-n matrix to compare distances between neighbors
    neighbors = np.zeros((len(unique_labels), len(unique_labels)))
    # iterate through neighbors and calculate distances
    for i in unique_labels:
        for j in superpixel_neighbors(superpixels, i):
            # only compute distances one way
            if j < i:
                neighbors[i, j] = metric(pixels[i], pixels[j])
    # fill out the other half of the matrix
    neighbors = neighbors + np.transpose(neighbors)
    # normalize to be within zero and one
    neighbors /= np.max(neighbors)
    # bigger values for more similarity
    neighbors[np.nonzero(neighbors)] = 1 - neighbors[np.nonzero(neighbors)]
    # FIXME this should be a sparse matrix
    return neighbors


def superpixel_neighbors(superpixels: np.ndarray, i: int):
    """Given a mapping from pixel location to superpixel label and a specific superpixel number, return the labels of all of the superpixels which neighbor the one specified."""
    selection = superpixels == i
    expanded = binary_dilation(selection)
    # binary XOR to get the difference
    difference = expanded ^ selection

    labels = set()
    for pixel in np.transpose(np.where(difference)):
        label = superpixels[tuple(pixel)]
        labels.add(label)
    return labels


def connected_within_threshold(superpixels: np.ndarray, distances: np.ndarray, delta: float = 0.01):
    """Given a mapping from pixel location to superpixel label as well as a weighted adjacency matrix, calculate the sets of connected components of a weighed undirected graph of superpixels "distances" whose weights are within a threshold delta, and return a newly labelled image."""
    # merged_labels maps index of node to a label for each newly merged group
    n, merged_labels = connected_components(distances < delta, directed=False)
    # superpixel_labels gives the label for the n'th superpixel
    superpixel_labels = np.unique(np.ma.compressed(superpixels))
    # create labelled image shaped like superpixels but masking everything
    labels = np.ma.array(np.zeros_like(superpixels), mask=True)
    # set labels for each pixel for each superpixel
    for index, label in enumerate(merged_labels):
        labels[superpixels == superpixel_labels[index]] = label
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


if __name__ == "__main__":
    main()