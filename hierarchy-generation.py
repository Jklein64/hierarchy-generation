from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor

def main():
    parser = ArgumentParser(description="Generate a segmentation hierarchy from an image.")
    parser.add_argument("image", help="Image file")
    parser.add_argument("config", help="Segmentation model config file")
    parser.add_argument("checkpoints", help="Segmentation model checkpoints file")
    args = parser.parse_args()

    print(args)

if __name__ == "__main__":
    main()