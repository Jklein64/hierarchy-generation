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

    print(wasserstein_image_distance(superpixels[135], superpixels[136]))

def wasserstein_image_distance(pixels_1: np.ndarray, pixels_2: np.ndarray) -> float:
    """Compute the Wasserstein or Earth Mover's distance between the given sets of pixels.  This function does not care what format the pixels are specified in, as long as they have r, g, and b components, but the format will affect whether or not you can compare outputs."""
    # remove transparent pixels and then alpha channel, then cast to float64
    pixels_1 = pixels_1[pixels_1[..., -1] != 0][..., 0:3].astype(np.float64)
    pixels_2 = pixels_2[pixels_2[..., -1] != 0][..., 0:3].astype(np.float64)
    # create weights and distance matrix
    weights_1 = np.ones(len(pixels_1)) / len(pixels_1)
    weights_2 = np.ones(len(pixels_2)) / len(pixels_2)
    distance = ot.dist(pixels_1, pixels_2)
    # find optimal flow
    optimal_flow = ot.lp.emd(weights_1, weights_2, distance, numItermax=300000)
    # derive Wasserstein distance
    return np.sum(optimal_flow * distance)




if __name__ == "__main__":
    main()