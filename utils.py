# project utilities

import numpy as np
import struct
import matplotlib.pyplot as plt
from array import array

def load_image(path):
    '''
    This function reads an IDX train/test file images and returns the reshaped image 
    (m x 28 x 28 x 1) and normalized [0 - 1] NumPy array.
    ---------------------------------------------------------------------------------------
    aguments:
            path: (str) IDX file path
    return:
            reshape_img: np array of size (m x 28 x 28 x 1)
    ---------------------------------------------------------------------------------------
    '''
    with open(path, 'rb') as img_data:
        magic, num, rows, cols = struct.unpack(">IIII", img_data.read(16))
        images = np.frombuffer(img_data.read(), dtype = np.uint8).reshape(num, rows, cols)
        reshape_img = images.reshape(-1, 28, 28, 1) / 255
        return reshape_img


def load_label(path):
    '''
    This function reads an IDX train/test file labels and returns the data as a NumPy array
    (m, 10) with 10 classes [0 - 9] and m examples.
    ----------------------------------------------------------------------------------------
    aguments:
            path: (str) IDX file path
    return:
            label: np array of size (m, 10)
    ---------------------------------------------------------------------------------------
    '''
    with open(path, 'rb') as lb_data:
        magic, num = struct.unpack(">II", lb_data.read(8))
        labels = np.array(array("B", lb_data.read()))
        label = np.eye(10, dtype = int)[labels]
        return label

def image_show(data, label, index):
    '''
    This function display the image and corresponding labels
    ---------------------------------------------------------------------------------------
    arguments:
            data: (features (ndarray)) train / test images to be displayed
            label: (array) corresponding label, either from prediction or normal label
            index: (int) specific example to be dispalyed
    return:
            the plot
    '''
    plt.figure(figsize=(5, 5))
    plt.imshow(data[index], cmap = 'gray')
    plt.title(f'It is {label[index]}')
    plt.axis('off')
    plt.show()