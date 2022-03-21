# Hierarchy Generation

This repository is for code for the automatic generation of a hierarchical segmentation of an image, which is a portion of my research group's project. The idea is that each segmentation of the image divides it into a small number of regions (as opposed to many), each of which can be further divided.

## Motivation

The image editing algorithm we have developed, _LoCoPalettes_, relies on a hierarchy of semantic segments in order to allow for edits to more and more specific regions of the image. The hierarchy is recursive in nature, where each parent node's region is the union of all of the child nodes' regions, so leaf nodes are the smallest and most specific segments, and the root node is the entire image. In all of our examples, we assembled the hierarchy by hand, stitching together and joining segments either from a single run of a neural network-based semantic segmentation algorithm or from manual tracing in Photoshop. However, _LoCoPalettes_ cannot be integrated into any photo editing software until this hierarchy can be generated automatically.

Our general approach to the automatic hierarchy generation process takes advantage of the recursive nature of the hierarchy. By paralleling this with a recursive segmentation, we can create a tree while avoiding the need to piece segments together after each segmentation. It also allows us to lazily segment the image, which will give the user more flexibility in editing, since they aren’t limited to the granularity of the original segmentation.

## Story

I began by experimenting with feeding just one segment from the output of a segmentation back in as input. I ran [SETR](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/setr/README.md) on an image, created a unique output image for each identified segment, and then selected one of those and attempted to run SETR on it again. As visible in the images below, the segmentation didn't reveal any more detail. I expected, for example, for the railing to be separated from the wall, since the colors are different, but that didn't happen.

| ![](images/uncropped-initial.png) | ![](images/uncropped-segment.png) |
| :-------------------------------: | :-------------------------------: |
|          initial segment          | result of segmentation of segment |

In order to identify these more detailed features, I experimented with isolating the initial segments into contiguous regions before saving them, and then cropping each result of the initial segmentation so that the segment takes up most of the image. I hypothesized that more specific semantic segments will be created if the area of interest took up a larger portion of the input. I also experimented with feeding back the part of the original image withing this bounding box, instead of the part of the segment. As visible in the images below, neither of these approaches really worked. While the segmentation of the cropped original image identified the railing as a distinct segment, other experimentation revealed that it generally doesn't indentify any detail beyond that of the initial segmentation.

| ![](images/setr-crop-initial.jpeg)  |   ![](images/setr-crop-view.png)   | ![](images/setr-crop-original.png) |
| :---------------------------------: | :--------------------------------: | :--------------------------------: |
| initial segment within bounding box | original image within bounding box |    segmented region of original    |

To continue testing my hypothesis that operating on a view of the original image will create a more detailed semantic segmentation, I attempted the same workflow but with [SenFormer](https://github.com/WalBouss/SenFormer), an ensemble learning-based network, instead of SETR, a transformer-based network. As visible in the images below, SenFormer struggles similarly to extract more detail, even from a cropped region of the original. Based on this experiment, I think that neural network-based image segmentation algorithms are going to be invariant to cropping (in addition to scaling and translation) of the input image, since these transforms are common when augmenting the training data.

| ![](images/senformer-segment.png) | ![](images/senformer-black.png) | ![](images/senformer-view.png) | ![](images/senformer-result.png) |
| :-------------------------------: | :-----------------------------: | :----------------------------: | :------------------------------: |
|     initial segment (cropped)     |     segmentation of initial     |   view of original (cropped)   |       segmentation of view       |

_currently, I am experimenting with non-network-based semantic segmentation. I am interested in merging the results of a superpixels segmentation, which doesn't require a network, in order to take advantage of the segmentation's edge-accuracy while accounting for oversegmentation. See [this journal entry about my own brainstorming](https://iron-salesman-ddf.notion.site/Think-about-Superpixels-approach-21ddce7bfbb5412e8c28ba8cf17b7349) or [this one about Yotam's suggestions](https://iron-salesman-ddf.notion.site/Think-about-Yotam-s-suggestion-096567b30d9f48b19b650bc91789e1f1) for more information about possible approaches._

# Setup

- Install [anaconda](https://www.anaconda.com/products/individual)/[miniconda](https://docs.conda.io/en/latest/miniconda.html), then create and activate the conda environment.

  ```bash
  conda env create -f environment.yml
  conda activate hierarchy-generation
  ```

- Download the [MMSegmentation repository](https://github.com/open-mmlab/mmsegmentation), which we need for the model configs.

  ```bash
  git clone git@github.com:open-mmlab/mmsegmentation.git
  ```

  Since this code makes inferences on the CPU, you will need to replace all instances of the string "SyncBN" in `configs/setr/setr_mla_512x512_160k_b8_ade20k.py` and `configs/_base_/models/setr_mla.py` with "BN" ([more info](https://github.com/open-mmlab/mmsegmentation/issues/292)).

- Download the checkpoints file (SETR-MLA trained on ADE20k from the [MMSegmentation Model Zoo](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/setr/README.md)).

  ```bash
  curl --output checkpoints.pth https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b8_ade20k/setr_mla_512x512_160k_b8_ade20k_20210619_191118-c6d21df0.pth
  ```

- Run the Python script.
  ```bash
  python heirarchy-generation.py "./image.png" "./mmsegmentation/configs/setr/setr_mla_512x512_160k_b8_ade20k.py" "./checkpoints.pth"
  ```
