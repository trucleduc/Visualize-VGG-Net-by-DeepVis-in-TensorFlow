# Visualize-VGG-Net-by-DeepVis-in-TensorFlow

## Introduction
I write this visualization of VGG-Net (http://www.robots.ox.ac.uk/~vgg/research/very_deep/) just to be familiar with TensorFlow. The method is exactly the same as DeepVis (http://yosinski.com/deepvis).

## How to use it
1. You need both Caffe (with Python interface) and TensorFlow installed. I know that there is TensorFlow version of VGG-Net, but I want to import it from Caffe for practice purpose.
2. Download (16-layer) VGG-Net (VGG_ILSVRC_16_layers_deploy.prototxt) and its pretrained weights (VGG_ILSVRC_16_layers.caffemodel)
3. Run "python convert_VGG.py" to convert VGG-Net from Caffe to TensorFlow format.
4. Run "python visualize_VGG.py" to produce per-class images (total 1000 images for 1000 ImageNet categories) that the network wants to see.

## Examples of a few per-class images
