from __future__ import annotations

from argparse import ArgumentParser

from scipy.sparse.csgraph import connected_components
from skimage.segmentation import slic
from time import perf_counter
from collections import deque
from PIL import Image
from cv2 import cv2

import scipy.io as sio
import numpy as np
import json

from metrics import Metric, ColorFeatures, FeaturesPCA
from visualization import save, show


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

    show(original, constraints=constraints)

    # load and set features for the image
    image_name = args.image[args.image.rfind("/")+1:args.image.rfind(".")]
    features = sio.loadmat(f"features/output/{image_name}.mat")['embedmap']
    
    start = perf_counter()
    labels, distances = precomputation(original, features)
    end = perf_counter()
    print(f"took {end-start} seconds to do precomputation")

    show(original, regions=labels, constraints=constraints)

    start = perf_counter()
    hierarchy, divided = generate_hierarchy(labels, distances, constraints)
    end = perf_counter()
    print(f"took {end-start} seconds to create the hierarchy")
    print(f"the hierarchy is: {hierarchy}")
    

    show(original, regions=divided[-1], constraints=constraints)

    # Smooth the segment with the most recent constraint
    for constraint in constraints:
        mask = (divided[-1] == divided[-1][constraint]).astype(np.uint8) * 255
        guided = cv2.ximgproc.guidedFilter(original, mask, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPSILON)
        show(np.insert(original[..., 0:3], 3, guided, axis=-1))

    def filter(*constraints):
        from functools import reduce
        d = divided[-1]
        m1 = reduce(lambda a, b: (d == a) | (d == b), constraints, (d == constraints[0]))
        m2 = m1.astype(np.uint8) * 255
        guided =  cv2.ximgproc.guidedFilter(original, m2, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPSILON)
        return np.insert(original[..., 0:3], 3, guided, axis=-1)

    # export files
    save(guided, filename="output/mask.png")
    np.save("output/regions", divided[-1])
    with open("output/hierarchy.json", "w") as file:
        json.dump(hierarchy, file)

    pass


def precomputation(image: np.ndarray, features: np.ndarray):
    """Create as much information as possible which can be persisted across hierarchy generations to save time.  This function doesn't depend on the constraints, so it can be run when an image is loaded into the GUI and then its values can be used later.  Features come from running the code in `features/`."""
    masked = (
        # mask transparent pixels if there's an alpha channel
        image[..., -1] == 0 if np.shape(image)[-1] == 4 
        # otherwise don't mask anything
        else np.zeros(np.shape(image)[:2], dtype=int))
    # each superpixel should have around 1,000 pixels
    n = len(np.transpose(np.where(~masked))) // 1000
    # labels maps pixel location to a superpixel label
    labels = np.array(slic(image[..., 0:3] / 255, n, start_label=0, mask=~masked))
    # create distances matrix comparing every pair of superpixels
    FeaturesPCA.set_features(features)
    distances = distances_matrix(image, labels, metric=ColorFeatures)
    return labels, distances


def generate_hierarchy(superpixels: np.ndarray, distances: np.ndarray, constraints: list[tuple[int, int]]):
    """Given the initial superpixels, precomputed distance matrix, and constraint locations sorted in order from least recently changed to modify its region to most recently changed, generate and return a hierarchy.  This function returns a nested Python dict `hierarchy` and a list of numpy arrays `divided`.  The array maps each pixel to its region's constraint at different levels, and the hierarchy describes how the regions merge together.  A key mapping to `None` is a leaf node.  Otherwise, key `k` maps to a dict which describes how the region belonging to `k` is divided with the addition of another constraint.  Note that `divided[-1] == k` yields a binary mask for the part of `k` which doesn't belong to any other constraints after accounting for all of them."""
    # create first level; root is implied
    hierarchy = { 0: None, 1: None }
    # compare between constraints 0 and 1, creating the first level of the hierarchy
    divided = [constrained_division(superpixels, np.where(superpixels < 0, -1, 0), distances, (0, 1), constraints)]

    for new_constraint, location in enumerate(constraints):
        # ignore the first two constraints
        if new_constraint < 2:
            continue
        # mask is True where the pixel is shared by this constraint
        # and the most specific constraint currently governing its pixel
        shared_constraint = divided[-1][location]
        shared_mask = divided[-1] == shared_constraint
        shared_superpixels = np.copy(superpixels)
        shared_superpixels[~shared_mask] = -1
        # remove non-shared ones from distances
        keep_labels = np.unique(shared_superpixels[shared_superpixels >= 0])
        # use set subtraction to maintain invariant with d
        yeet_labels = set(np.unique(superpixels[superpixels >= 0])) - set(keep_labels)
        yeet_mask = np.ones_like(distances).astype(bool)
        for i in yeet_labels:
            # remove i'th row and column
            yeet_mask[i, ...] = False
            yeet_mask[..., i] = False
        d = len(keep_labels)
        # recreate distances matrix after removing those superpixels
        shared_distances = np.reshape(distances[yeet_mask], (d, d))
        divided.append(constrained_division(
            superpixels=shared_superpixels, 
            previous=divided[-1], 
            distances=shared_distances, 
            c_i=(shared_constraint, new_constraint), 
            constraints=constraints))
        queue = deque([[0]])
        # update hierarchy with BFS
        while len(queue) > 0:
            level = [queue.popleft() for _ in range(len(queue))]
            # node is a list of constraints into the hierarchy
            for node in level:
                if node is not None:
                    # wrap so view[0] is root
                    view = [hierarchy]
                    for i in node:
                        view = view[i]
                    if shared_constraint in view:
                        if view[shared_constraint] is None:
                            view[shared_constraint] = dict()
                        # updating view updates hierarchy since
                        # its a reference to the same object
                        view[shared_constraint].update({
                            int(shared_constraint): None,
                            int(new_constraint): None})
                        break
                    if view is not None:
                        for child in list(view.keys()):
                            queue.append([*node, child])


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
    # use -2 to avoid conflicts with -1 as masked
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