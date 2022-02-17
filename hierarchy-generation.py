from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor

def main():
    parser = ArgumentParser(description="Generate a segmentation hierarchy from an image.")
    parser.add_argument("image", help="Image file")
    parser.add_argument("config", help="Segmentation model config file")
    parser.add_argument("checkpoints", help="Segmentation model checkpoints file")
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    # Figure out how to use CUDA later, if it's even possible on M1
    model = init_segmentor(args.config, args.checkpoints, device="cpu")

    result = inference_segmentor(model, args.image)

    print(result)

if __name__ == "__main__":
    main()