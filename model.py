# Implement lenet-5 model

import numpy as np
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfl

def lenet_model(input_shape):
    '''
    Implement Functional Keras API to train model. The model has 5 layers
    -----------------------------------------------------------------------------------
    1. conv layer:
            k_size: (5, 5)
            stride: (1, 1)
            n_filters: 6
            padding: 'same'
            activation: relu
            
    2. avg pooling:
            pool_size: (2, 2)
            
    3. conv layer:
            k_size: (5, 5)
            stride: (1, 1)
            n_filters: 16
            padding: 'valid'
            activation: relu
            
    4. avg pooling
            pool_size: (2, 2)
            strides: (2, 2)

    5. conv layer:
            k_size: (5, 5)
            stride: (1, 1)
            n_filters: 120
            padding: 'valid'
            activation: relu
            
    5. flatten
    6. classifier
            activation: softmax
            classes: 10
    -----------------------------------------------------------------------------------------------------------------------
    '''
    input_img = tfk.Input(shape = input_shape)

    z1 = tfl.Conv2D(kernel_size = (5, 5), strides = (1, 1), filters = 6, padding = 'same', activation = 'tanh')(input_img)
    p1 = tfl.AveragePooling2D(pool_size = (2, 2))(z1)
    z2 = tfl.Conv2D(kernel_size = (5, 5), strides = (1, 1), filters = 16, padding = 'valid', activation = 'tanh')(p1)
    p2 = tfl.AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(z2)
    z3 = tfl.Conv2D(kernel_size = (5, 5), strides = (1, 1), filters = 120, padding = 'valid', activation = 'tanh')(p2)
    flat = tfl.Flatten()(z3)
    tanh = tfl.Dense(units = 84, activation = 'tanh')(flat)
    output = tfl.Dense(units = 10, activation = 'softmax')(tanh)

    model = tfk.Model(inputs = input_img, outputs = output)
    return model