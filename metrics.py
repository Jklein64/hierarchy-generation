from __future__ import annotations

import numpy as np
import ot

class Metric:
    """Class wrapper for metrics, used to precompute parts for efficiency."""

    def __init__(self, pixels: np.ndarray) -> None:
        self.pixels = pixels
        self.value = self.compute()

    def compute(pixels: np.ndarray):
        pass

    @staticmethod
    def compare(a: Metric, b: Metric):
        pass


class AverageColor(Metric):
    def compute(self):
        return np.mean(self.pixels, axis=0)

    def compare(a, b):
        return sum(np.square(a.value - b.value))


class Wasserstein(Metric):
    def compute(self):
        n = len(self.pixels)
        r, g, b = self.color_histograms()
        return (r / n, g / n, b / n)

    def compare(a, b):
        red_1, green_1, blue_1 = a.value
        red_2, green_2, blue_2 = b.value
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


# make something semantic??

