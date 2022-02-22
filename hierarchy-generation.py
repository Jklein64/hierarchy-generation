from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor
from PIL import Image
import numpy as np


def main():
    parser = ArgumentParser(description="Generate a segmentation hierarchy from an image.")
    parser.add_argument("image", help="Image file")
    parser.add_argument("config", help="Segmentation model config file")
    parser.add_argument("checkpoints", help="Segmentation model checkpoints file")
    args = parser.parse_args()

    segments = segment_image(np.array(Image.open(args.image)), args.config, args.checkpoints)
    for i, segment in enumerate(segments):
        Image.fromarray(segment).save(f"output/segment-{i}.png")


def segment_image(image, config, checkpoints):
    """Segment the given image with the segmentation model described by config and checkpoints.  Expects image to be an array and the others to be file paths.  Returns a list of image-sized arrays, one for each segment.
    """
    # build the model from a config file and a checkpoint file
    # Figure out how to use CUDA later, if it's even possible on M1
    model = init_segmentor(config, checkpoints, device="cpu")
    # result has shape 1 x H x W, storing an integer label
    result = inference_segmentor(model, image[:,:,0:3])[0]
    # add a new axis to the image for alpha. Roll is needed to add to the end
    image = np.roll(np.insert(image[:,:,0:3], 0, 255, axis=-1), -1, axis=-1)
    
    segments = []
    for label in np.unique(result):
        # make new image with same dimensions
        output = np.zeros_like(image)
        # set pixels to result where label is label
        np.copyto(output, image, where=np.expand_dims(result == label, axis=-1))
        # collect into list of segments
        segments.append(output)

    return segments        


if __name__ == "__main__":
    main()