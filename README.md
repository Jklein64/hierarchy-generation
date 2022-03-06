# Hierarchy Generation

This repository is for code for the automatic generation of a hierarchical segmentation of an image. The idea is that each segmentation of the image divides it into a small number of regions (as opposed to many), each of which can be further divided if needed.

The recursive nature of the algorithm means that we create a hierarchy as we segment, and don’t need to piece it together after segmentation. It also allows us to lazily segment the image, which will give the user more flexibility in editing (since they aren’t limited to the granularity of the original segmentation). This method can also integrate some aspects of segment pairing.

# Steps

- [ ] Find a method to perform non-rectangular segmentation ([Slack discussion](https://adoberesearch.slack.com/archives/C02FSA4Q746/p1641707240001600))
- Solid black/white fill in parts of the bounding box that aren’t part of the region?
- Something creative involving farthest color from bordering pixels?
- [ ] Store regions in a tree and compute recursively; test with file/folder output
- [ ] Integrate with Python GUI ([GitHub repo](https://github.com/tedchao/Sparse-editing))

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
