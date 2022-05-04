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

Having determined that neural network-based segmentation cannot perform recursive segmentation, I decided to segment with [superpixels](https://pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/), which don't depend on a neural network. Superpixels oversegment the image, so I compute the similarity between every pair of superpixels using the Wasserstein distance and then combine sufficiently similar superpixels into non-contiguous regions. This worked, but took a long time to compute (around four seconds) and, in a different test case, merged superpixels which were part of different semantic regions that happened to have similar color distributions.

The images below illustrate the iterative merging of the superpixels for different values of delta, the similarity threshold. Delta allows for interpolation between merging none of the superpixels and all of the superpixels. Images with larger delta values have more general regions, each of which are a union of some of the regions in the image with a smaller delta value. Sweeping values for delta creates a merging schedule that we can turn into a hierarchy.

| ![](images/superpixels-0.00.png) | ![](images/superpixels-0.01.png) | ![](images/superpixels-0.02.png) | ![](images/superpixels-0.05.png) |
| :------------------------------: | :------------------------------: | :------------------------------: | :------------------------------: |
|   most detailed; delta = 0.00    |   more detailed; delta = 0.01    |    more general; delta = 0.02    |    most general; delta = 0.05    |

Finding an optimal value for delta without any additional context is actually and extremely difficult problem. Fortunately, I have additional context: constraints. The goal of segmentation with two constraints is to divide the original region into two sub-regions, each of which contain only one constraint. This means that I want to find the largest value of delta for which the constraints are still part of separate regions, which can be found with a simple binary search.

The images below demonstrate the result of this delta optimization process. Here, the user might want to recolor the ground on the bottom while making no changes to the ground on the top of the image. While the segmentation was able to separate the two regions, it also grouped the baseball player's arm together with the ground, since the pixels shared enough colors.

| ![](images/baseball-original.png) | ![](images/baseball-superpixels.png) | ![](images/baseball-unmerged.png) | ![](images/baseball-merged.png) |
| :-------------------------------: | :----------------------------------: | :-------------------------------: | :-----------------------------: |
|          original image           |             superpixels              |          optimized delta          |     merged to show regions      |

_currently, I am experimenting with variations on the distance metric and possibly merging neighboring superpixels before computing pairwise distances across all of them to increase performance. I am also looking into integration of [YOLO](https://github.com/ultralytics/yolov5) or another semantic identification or segmentation algorithm to identify superpixels corresponding to objects in the image that need to be kept together as a group and not merged with others._

# Setup

- Install [anaconda](https://www.anaconda.com/products/individual)/[miniconda](https://docs.conda.io/en/latest/miniconda.html), then create and activate the conda environment.

  ```bash
  conda env create -f environment.yml
  conda activate hierarchy-generation
  ```

- Run the python script. See the docstring for more info.

  ```bash
  python hierarchy-generation.py -h
  ```
