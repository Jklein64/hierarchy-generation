from __future__ import annotations

from argparse import ArgumentParser

from scipy.sparse.csgraph import connected_components
from skimage.segmentation import slic
from collections import deque
from PIL import Image
from cv2 import cv2

import numpy as np

from metrics import Metric, ColorFeatures
from visualization import show


# determined through experimentation
GUIDED_FILTER_RADIUS = 30
# taken from example on GitHub
GUIDED_FILTER_EPSILON = 1e-6 * 255 ** 2
# picked randomly and it happened to work
DELTA_OPTIMIZATION_THRESHOLD = 0.001


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

    masked = (
        # mask transparent pixels if there's an alpha channel
        original[..., -1] == 0 if np.shape(original)[-1] == 4 
        # otherwise don't mask anything
        else np.zeros(np.shape(original)[:2], dtype=int))
    # each superpixel should have around 1,000 pixels
    n = len(np.transpose(np.where(~masked))) // 1000
    # labels maps pixel location to a superpixel label
    labels = np.array(slic(original[..., 0:3] / 255, n, start_label=0, mask=~masked))

    # show the initial superpixel segmentation
    show(original, regions=labels, constraints=constraints)
    
    # create dense distances matrix and merge based on optimized delta
    distances = distances_matrix(original, labels, metric=ColorFeatures)

    # store constraint assignment at each level
    divided = list()

    # compare between constraints 0 and 1, creating the first level of the hierarchy
    divided.append(constrained_division(labels, np.where(labels < 0, -1, 0), distances, (0, 1), constraints))

    # create root node and first level
    hierarchy = { 0: None, 1: None }

    # show the image after applying the first two constraints
    show(original, regions=divided[0], constraints=constraints)

    for new_constraint, location in enumerate(constraints):
        # ignore the first two constraints
        if new_constraint < 2:
            continue
        # mask is True where the pixel is shared by this constraint
        # and the most specific constraint currently governing its pixel
        shared_constraint = divided[-1][location]
        shared_mask = divided[-1] == shared_constraint
        shared_superpixels = np.copy(labels)
        shared_superpixels[~shared_mask] = -1
        # remove non-shared ones from distances
        keep_labels = np.unique(shared_superpixels[shared_superpixels >= 0])
        # use set subtraction to maintain invariant with d
        yeet_labels = set(np.unique(labels[labels >= 0])) - set(keep_labels)
        yeet_mask = np.ones_like(distances).astype(bool)
        for i in yeet_labels:
            # remove i'th row and column
            yeet_mask[i, ...] = False
            yeet_mask[..., i] = False
        d = len(keep_labels)
        # recreate distances matrix after removing those superpixels
        shared_distances = np.reshape(distances[yeet_mask], (d, d))
        # TODO use the tuple passed below to help make the hierarchy
        divided.append(constrained_division(
            superpixels=shared_superpixels, 
            previous=divided[-1], 
            distances=shared_distances, 
            c_i=(shared_constraint, new_constraint), 
            constraints=constraints))
        # update hierarchy with BFS
        queue = deque([ [0] ])
        while len(queue) > 0:
            level = [queue.popleft() for _ in range(len(queue))]
            for node in level:
                # wrap so view[0] is root
                view = [hierarchy]
                for i in node:
                    view = view[i]
                if shared_constraint in view:
                    if view[shared_constraint] is None:
                        view[shared_constraint] = dict()
                    # updating view updates hierarchy
                    # because python dicts are cool
                    view[shared_constraint].update({
                        shared_constraint: None,
                        new_constraint: None})
                    break
                # list gets actual values instead of view
                queue.extend(list(view[i].values()))

        # show the image after addressing the third constraint
        show(original, regions=divided[-1], constraints=constraints)

    # show the segments after applying a guided filter
    for label in np.unique(divided[-1][divided[-1] >= 0]):
        # cv2 doesn't like boolean arrays
        mask = (divided[-1] == label).astype(np.uint8) * 255
        refined = cv2.ximgproc.guidedFilter(original, mask, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPSILON)
        # use refined mask value as alpha channel on original image
        show(np.insert(original[..., 0:3], 3, refined, axis=-1))

    print(hierarchy)

    pass


def constrained_division(superpixels: np.ndarray, previous: np.ndarray, distances: np.ndarray, c_i: tuple[int, int], constraints: list):
    """Given a possibly-transparent image's masked superpixel segmentation, the previous assignment of constraints, the pairwise distance between those superpixels (after RAG merging), and two indices into the given constraints list, divide the image into additional semantic regions such that each constraint is in its own region.  The labels returned from this method correspond to the given constraints."""
    old_constraint = constraints[c_i[0]]
    new_constraint = constraints[c_i[1]]
    # track the index of the original label for each constraint
    constraint_labels = tuple(
        np.searchsorted(np.unique(superpixels[superpixels >= 0]), superpixels[constraint]) for
        constraint in (old_constraint, new_constraint))
    # binary search to find the largest value of 
    # delta which still separates the constraints
    low = 0
    high = np.max(distances)
    delta = (high + low) / 2
    # FIXME justify threshold choice
    threshold = high * DELTA_OPTIMIZATION_THRESHOLD
    # find which merged superpixel each constraint would belong to
    a, b = constraints_within_threshold(constraint_labels, distances < delta)
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
        a, b = constraints_within_threshold(constraint_labels, distances < delta)
    # actually create the labelled image from the connected components
    merged = connected_within_threshold(superpixels, distances < delta)
    # assign regions not containing any constraint to the older one
    constraint_labels = merged[tuple(np.transpose(constraints))]
    for label in np.unique(merged[merged >= 0]): 
        if label not in constraint_labels:
            # replace label with less recent constraint
            merged[merged == label] = a
    # make merged store constraint index at each pixel
    conditions = []
    replacements = []
    # +1 to include end of range, +1 again to include new constraint
    for i, c in enumerate(constraints[:np.max(previous) + 2]):
        label = merged[c]
        # ignore masked labels
        if label >= 0:
            conditions.append(merged == label)
            replacements.append(i)
    # use -1 as "masked" since masked arrays get overridden
    merged = np.select(conditions, replacements, default=-2)
    # fill masked values with previous constraint
    merged[merged == -2] = previous[merged == -2]
    return merged


def constraints_within_threshold(constraint_labels: tuple[int, int], adjacency: np.ndarray):
    """Given the label from the initial superpixel segmentation for each constraint and a boolean adjacency matrix marking edges between these superpixels, return the label of the region that each constraint would occupy if the superpixels were merged according to the connected components of the adjacency matrix.  This function *does not* compute labels for every pixel!"""
    # merged_labels maps index of node to a label for each newly merged group
    n, merged_labels = connected_components(adjacency, directed=False)
    return merged_labels[list(constraint_labels)]


def connected_within_threshold(superpixels: np.ndarray, adjacency: np.ndarray):
    """Given every pixel's label from the initial superpixel segmentation and a boolean adjacency matrix, compute, assign, and return labels which map each pixel to new superpixels which have been merged according the the adjacency matrix."""
    # merged_labels maps index of node to a label for each newly merged group
    n, merged_labels = connected_components(adjacency, directed=False)
    # superpixel_labels gives the label for the n'th superpixel
    superpixel_labels = np.unique(superpixels[superpixels >= 0])
    # create labelled image shaped like superpixels but masking everything
    labels = np.ones_like(superpixels) * -1
    # set labels for each pixel for each superpixel
    for index, label in enumerate(merged_labels):
        labels[superpixels == superpixel_labels[index]] = label
    return labels


def distances_matrix(original: np.ndarray, superpixels: np.ndarray, metric: Metric) -> np.ndarray:
    """Create a matrix with the metric-based distances between every pair of the given superpixels implied by the original and labelled images."""
    # store list of valid superpixel labels
    unique_labels = np.unique(superpixels[superpixels >= 0])
    # bundle index information with rgb
    precomputed = []
    for label in unique_labels:
        selection = superpixels == label
        i, j = np.where(selection)
        rgb = original[selection]
        rgbij = np.column_stack((rgb, i, j))
        precomputed.append(metric(rgbij))
    # create n-by-n matrix to compare distances between n superpixels
    distances = np.zeros((len(unique_labels), len(unique_labels)))
    # distance is symmetric, so only compare each pair once (below diagonal)
    for i, j in np.transpose(np.tril_indices(len(unique_labels), k=-1)):
        a = precomputed[i]
        b = precomputed[j]
        distances[i, j] = metric.compare(a, b)
    # fill in the rest of the distances matrix
    return distances + np.transpose(distances)


if __name__ == "__main__":
    main()