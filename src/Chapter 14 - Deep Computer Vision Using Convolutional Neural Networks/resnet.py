import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [  # the otherwise normal layers if we weren't using residual units
            Conv2D(filters, kernel_size=(3, 3), strides=strides,
                   padding="same", use_bias=False),
            BatchNormalization(),
            self.activation,
            Conv2D(filters, (3, 3), strides=(1, 1),
                   padding="same", use_bias=False),
            BatchNormalization()
        ]
        
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                Conv2D(filters, kernel_size=(1, 1), strides=strides,
                       padding="same", use_bias=False),
                BatchNormalization()
            ]
    
    def call(self, inputs):
        Z = inputs
        
        # propagate forward thru main layers
        for layer in self.main_layers:
            Z = layer(Z)
        
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        
        return self.activation(Z + skip_Z)