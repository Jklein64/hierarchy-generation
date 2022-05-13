from __future__ import annotations

import scipy.io as sio
import numpy as np
import ot

def pca(features: np.ndarray, dim = None):
    """Compute the PCA of the given feature vectors.  If the number of dimensions to project to isn't passed, then it will pick a number of vectors such that they can explain 95% of the variance."""
    width, height, depth = np.shape(features)
    vectors = np.reshape(features, (width * height, depth))
    # z-score and create covariance
    mu = np.mean(vectors, axis=0)
    sigma = np.std(vectors, axis=0)
    standardized = (vectors - mu) / sigma
    covariance = np.cov(standardized.T)
    # make basis of 3 most significant eigenvectors
    eigenstuffs = np.linalg.eig(covariance)
    # argsort is ascending but we want descending
    order = np.flip(np.argsort(eigenstuffs[0]))
    eigenvectors = eigenstuffs[1][..., order]
    # pick dim based on 95% rule
    if dim is None:
        eigenvalues = eigenstuffs[0][order]
        explained = eigenvalues / np.sum(eigenvalues)
        # +1 to include the one that pushes it over .95
        dim = np.flatnonzero(np.cumsum(explained) > .95)[0] + 1
    basis = eigenvectors[..., :dim]
    # project features onto basis
    projected = vectors @ basis
    # normalize to within [0, 1]
    projected -= np.min(projected, axis=0)
    projected /= np.max(projected, axis=0)
     # turn into 8-bit image
    image = np.reshape(projected, (width, height, dim))
    return (image * 255).astype(np.uint8)


# change this when changing the input image
features: np.ndarray = sio.loadmat("features/output/nesi.mat")['embedmap']
# project features to most significant dimensions
features_pca = pca(features)


class Metric:
    """Class wrapper for metrics, used to precompute parts for efficiency."""

    def __init__(self, rgbij: np.ndarray) -> None:
        self.rgbij = rgbij
        self.pixels: np.ndarray = rgbij[..., 0:3]
        self.indices: tuple[np.ndarray] = tuple(rgbij[..., 3:5].T)
        self.value = self.compute()

    def compute(self):
        pass

    def compare(self, other: Metric):
        pass


class AverageColor(Metric):
    def compute(self):
        return np.mean(self.pixels, axis=0)

    def compare(self, other):
        return np.linalg.norm(self.value - other.value)


class FeaturesPCA(Metric):
    def compute(self):
        return np.mean(features_pca[self.indices], axis=0)

    def compare(self, other: Metric):
        return np.linalg.norm(self.value - other.value)


class ColorFeatures(Metric):
    # importance of features
    gamma = 0.6

    def compute(self):
        rgb = AverageColor(self.rgbij)
        fea = FeaturesPCA(self.rgbij)
        return (rgb, fea)

    def compare(self, other: Metric):
        rgb_a, fea_a = self.value
        rgb_b, fea_b = other.value
        rgb_dist = rgb_a.compare(rgb_b)
        fea_dist = fea_a.compare(fea_b)
        gamma = ColorFeatures.gamma
        return (1 - gamma) * rgb_dist + gamma * fea_dist


class Wasserstein(Metric):
    def compute(self):
        n = len(self.pixels)
        r, g, b = self.color_histograms()
        return (r / n, g / n, b / n)

    def compare(self, other):
        red_1, green_1, blue_1 = self.value
        red_2, green_2, blue_2 = other.value
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

    def color_histograms(self) -> list[np.ndarray]:
        """Maps a list of rgb pixels with integer values to three frequency charts with 256 bins each, one for each color channel.  The frequencies are NOT normalized.  This function is used in the Wasserstein-based metrics."""
        r, g, b = np.transpose(self.pixels)[0:3]
        histograms = []
        for channel in (r, g, b):
            # need explicit integer type since np.float64 is default
            # needs to be big enough to store a count! uint8 won't do!
            histogram = np.zeros((2 ** 8), dtype=np.uint64)
            values, counts = np.unique(channel, return_counts=True)
            histogram[values] = counts
            histograms.append(histogram)
        return histograms


class CentroidDistance(Metric):
    def compute(self):
        return np.mean(np.transpose(self.indices), axis=0)

    def compare(self, other: Metric):
        return np.linalg.norm(self.value - other.value)