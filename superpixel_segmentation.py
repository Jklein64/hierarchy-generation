from __future__ import annotations
from skimage.segmentation import slic, mark_boundaries
from PIL import Image
import ot, numpy as np

def main():
    # given original, which is a segment, possibly with opacity
    original = np.array(Image.open("./output/segment-1.png"))
    # slic() expects images to be floating point and 3-channel
    image = original[..., 0:3] / 255
    # labels is an array which maps each pixel location to a segment label
    labels = slic(image, 200, start_label=0, multichannel=True)
    # superpixels is an array of pixels for each label
    superpixels = [original[labels == label] for label in np.unique(labels)]

    # create lower-triangular distances matrix
    # distances = np.zeros((len(superpixels), len(superpixels)))
    # for i in range(0, len(superpixels)):
    #     for j in range(0, i):
    #         ...

    # print(wasserstein_image_distance(superpixels[135], superpixels[136]))
    print(wasserstein_image_distance(superpixels[137], superpixels[138]))

def wasserstein_image_distance(pixels_1: np.ndarray, pixels_2: np.ndarray) -> float:
    """Compute the Wasserstein or Earth Mover's distance between the given sets of pixels.  This function does not care what format the pixels are specified in, as long as they have r, g, and b components, but the format will affect whether or not you can compare outputs."""
    # remove completely transparent pixels
    pixels_1 = pixels_1[pixels_1[..., -1] != 0]
    pixels_2 = pixels_2[pixels_2[..., -1] != 0]
    # early return for irrelevant superpixels to avoid divide by zero
    if len(pixels_1) == 0 and len(pixels_2) == 0:
        return 0
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
    print(wasserstein_red, wasserstein_green, wasserstein_blue)
    # average the channel-based distances to get final metric
    return sum([wasserstein_red, wasserstein_green, wasserstein_blue]) / 3


def color_histograms(pixels: np.ndarray) -> list[np.ndarray]:
    """Maps a list of rgb pixels with integer values to three frequency charts with 256 bins each, one for each color channel.  The frequencies are NOT normalized."""
    r, g, b = np.transpose(pixels)[0:3]
    histograms = []
    for channel in (r, g, b):
        # need explicit integer type since np.float64 is default
        # needs to be big enough to store a count! uint8 won't do!
        histogram = np.zeros((2 ** 8), dtype=np.uint64)
        values, counts = np.unique(channel, return_counts=True)
        # there's someting fishy here
        histogram[values] = counts
        histograms.append(histogram)
    return histograms


if __name__ == "__main__":
    main()