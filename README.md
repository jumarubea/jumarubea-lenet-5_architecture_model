# lenet-5_architecture_model
# Handwritten Digit Recognition

This project implements the LeNet-5 neural network architecture to recognize handwritten digits using the MNIST dataset.


## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

LeNet-5 is a classic convolutional neural network (CNN) architecture designed by Yann LeCun,[learn more](https://en.wikipedia.org/wiki/LeNet), primarily for handwritten digit classification. This project uses [TensorFlow](https://www.tensorflow.org/guide/keras/functional_api) and Keras to build and train the LeNet-5 model on the MNIST dataset. The dataset is already included in this project.


## Prerequisites

Make sure you have the following installed:

- [Python_3.6+] (https://www.python.org/downloads/)
- [TensorFlow_2.x] (https://www.tensorflow.org/install) 
- [NumPy] (https://numpy.org/install/)
- [Matplotlib](https://matplotlib.org/stable/install/index.html)
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) (optional)

## Installation

Clone this repository:

```bash
git clone https://github.com/jumarubea/lenet-5_architecture_model.git
cd lenet-5-digit-recognition
```
## Model Architecture
LeNet-5 consists of the following layers:

- Convolutional Layer: 6 filters of size 5x5, activation function: tanh
- Average Pooling Layer: pool size 2x2
- Convolutional Layer: 16 filters of size 5x5, activation function: tanh
- Average Pooling Layer: pool size 2x2
- Convolutional Layer: 120 filters of size 5x5, activation function: tanh
- Flatten Layer
- Dense Layer: 84 units, activation function: tanh
- Output Layer: 10 units, activation function: softmax

Note: for the purpose of accuracy measure, i implement `relu` activation instead of `tanh`
    except for the 84 dense layer.

## Results
The trained LeNet-5 model achieves a test accuracy of approximately 98% on the MNIST dataset.

## Contributing
If you want to contribute to this project, please fork the repository and submit a pull request.
