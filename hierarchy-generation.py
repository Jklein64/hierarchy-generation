from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor
from scipy.ndimage import label
from PIL import Image
import numpy as np


def main():
    parser = ArgumentParser(description="Generate a segmentation hierarchy from an image.")
    parser.add_argument("image", help="Image file")
    parser.add_argument("config", help="Segmentation model config file")
    parser.add_argument("checkpoints", help="Segmentation model checkpoints file")
    args = parser.parse_args()

    segments = segment_image(np.array(Image.open(args.image)), args.config, args.checkpoints)
    contiguous_segments = []

    for segment in segments:
        # label() expects a 2D array, so we flatten by summing
        labeled, _ = label(np.sum(segment, axis=-1))
        # get all of the contiguous regions of the segment
        contiguous_subsegments = separate_by_label(segment, labeled)
        # collect the contiguous regions
        for contigous_subsegment in contiguous_subsegments:
            contiguous_segments.append(contigous_subsegment)
    
    for i, s in enumerate(contiguous_segments):
        Image.fromarray(s).save(f"output/segment-{i}.png")


def segment_image(image, config, checkpoints):
    """Segment the given image with the segmentation model described by config and checkpoints.  Expects image to be an array and the others to be file paths.  Returns a list of image-sized arrays, one for each segment."""
    # build the model from a config file and a checkpoint file
    # Figure out how to use CUDA later, if it's even possible on M1
    model = init_segmentor(config, checkpoints, device="cpu")
    # result has shape 1 x H x W, storing an integer label
    result = inference_segmentor(model, image[:,:,0:3])[0]
    # add a new axis to the image for alpha. Roll is needed to add to the end
    image = np.roll(np.insert(image[:,:,0:3], 0, 255, axis=-1), -1, axis=-1)
    # separate the image into a list of image, split by result
    return separate_by_label(image, result) 


def separate_by_label(array, label_array):
    """Separate the given array into a list of similarly-shaped arrays, one for each unique label in the given array of labels."""
    segments = list()
    # iterate across possible label values
    for label in np.unique(label_array):
        # make new array with same dimensions
        output = np.zeros_like(array)
        # copy values from array to output when the label at that position matches label
        np.copyto(output, array, where=np.expand_dims(label_array == label, axis=-1))
         # collect into list of segments
        segments.append(output)

    return segments


if __name__ == "__main__":
    main()