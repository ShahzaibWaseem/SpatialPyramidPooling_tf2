# Spatial Pyramid Pooling

Spatial pyramid pooling layers for keras, based on the paper [Spatial Pyramid Pooling in Deep Convolutional
Networks for Visual Recognition](https://arxiv.org/abs/1406.4729).

I used this starter code, which is in tensorflow 1.0, [github.com/yhenon/keras-spp](https://github.com/yhenon/keras-spp). This code is for tensorflow 2.0.

![spp](http://i.imgur.com/SQWJVoD.png)

## When to use

Apply the pooling procedure on the entire image, given an image batch. This is especially useful if the image input can have varying dimensions, but needs to be fed to a fully connected layer.

## Configuration

- Python 3.6
- Tensorflow 2.0
- Keras 2.2.4
- Tensorflow input shape was used (not Theano)

### Input Ordering

This code supports the Thaeno version if you change the input order
Tensorflow input order: `(samples, rows, cols, channels)` (image_data_format='channels_last').
Theano input order: `(samples, channels, rows, cols)` (image_data_format='channels_first').
