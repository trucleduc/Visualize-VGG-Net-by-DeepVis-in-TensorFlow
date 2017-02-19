# Visualize-VGG-Net-by-DeepVis-in-TensorFlow

## Introduction
I write this visualization of VGG-Net (http://www.robots.ox.ac.uk/~vgg/research/very_deep/) just to be familiar with TensorFlow. The method is exactly the same as DeepVis (http://yosinski.com/deepvis).

## How to use it
1. You need both Caffe (with Python interface) and TensorFlow installed. I know that there is TensorFlow version of VGG-Net, but I want to import it from Caffe for practicing purpose.
2. Download (16-layer) VGG-Net (VGG_ILSVRC_16_layers_deploy.prototxt) and its pretrained weights (VGG_ILSVRC_16_layers.caffemodel)
3. Run "python convert_VGG.py" to convert VGG-Net from Caffe to TensorFlow format.
4. Run "python visualize_VGG.py" to produce per-class images (total 1000 images for 1000 ImageNet categories) that the network wants to see.

## Examples of a few per-class images
![.](/per-class-images/prob-000.png)
![.](/per-class-images/prob-001.png)
![.](/per-class-images/prob-002.png)
![.](/per-class-images/prob-003.png)
![.](/per-class-images/prob-004.png)
![.](/per-class-images/prob-005.png)
![.](/per-class-images/prob-006.png)
![.](/per-class-images/prob-007.png)
![.](/per-class-images/prob-008.png)
![.](/per-class-images/prob-009.png)
![.](/per-class-images/prob-010.png)
![.](/per-class-images/prob-011.png)
![.](/per-class-images/prob-012.png)
![.](/per-class-images/prob-013.png)
![.](/per-class-images/prob-014.png)
![.](/per-class-images/prob-015.png)
![.](/per-class-images/prob-016.png)
![.](/per-class-images/prob-017.png)
![.](/per-class-images/prob-018.png)
![.](/per-class-images/prob-019.png)
![.](/per-class-images/prob-020.png)
![.](/per-class-images/prob-021.png)
![.](/per-class-images/prob-022.png)
![.](/per-class-images/prob-023.png)
![.](/per-class-images/prob-024.png)
![.](/per-class-images/prob-025.png)
![.](/per-class-images/prob-026.png)
![.](/per-class-images/prob-030.png)
![.](/per-class-images/prob-031.png)
![.](/per-class-images/prob-032.png)
![.](/per-class-images/prob-033.png)
![.](/per-class-images/prob-034.png)
![.](/per-class-images/prob-035.png)
![.](/per-class-images/prob-036.png)
![.](/per-class-images/prob-037.png)
![.](/per-class-images/prob-038.png)
![.](/per-class-images/prob-039.png)
![.](/per-class-images/prob-040.png)
![.](/per-class-images/prob-041.png)
![.](/per-class-images/prob-042.png)
![.](/per-class-images/prob-043.png)
![.](/per-class-images/prob-044.png)
![.](/per-class-images/prob-045.png)
