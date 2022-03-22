from skimage.segmentation import slic, mark_boundaries
from PIL import Image
import numpy as np

# slic() expects images to be floating point and 3-channel
image = np.array(Image.open("./output/segment-1.png"))[...,0:3] / 255
# segments is an array which maps each pixel location to a segment label
segments = slic(image, 200, start_label=0, multichannel=True)
Image.fromarray((mark_boundaries(image, segments) * 255).astype(np.uint8)).show()
pass