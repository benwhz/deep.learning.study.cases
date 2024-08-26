import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import ops
from keras import layers

class Linear(layers.Layer):
    def __init__(self, units = 32, input_dim = 32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b
    
x = ops.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y, type(y))

assert linear_layer.weights == [linear_layer.w, linear_layer.b]