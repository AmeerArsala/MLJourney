from tensorflow import keras
from keras import layers
from keras.layers import Dense, Activation, BatchNormalization

class FunnelModel(keras.Model):
    def __init__(self, hidden_layers=[], output_layers=[], **kwargs):
        super().__init__(**kwargs)
        
        def compose(args):
            return Dense(units=args[0], activation=args[1])
        
        self.hidden_layers_start = [
            compose(hidden_layers[0]),
            compose(hidden_layers[0]),
            compose(hidden_layers[0])
        ]
        
        self.hidden_layers_rest = [
            compose(hidden_layers[1]),
            compose(hidden_layers[2]),
            compose(hidden_layers[3])
        ]
        
        self.output_layers = [compose(output_layers[0])]
    
    
    def call(self, inputs):
        (inputA, inputB) = inputs
        
        hl1_A = self.hidden_layers_start[0](inputA)
        hl1_B = self.hidden_layers_start[1](inputB)
        hl1_AB = self.hidden_layers_start[2](layers.concatenate([inputA, inputB]))
        
        hl2 = self.hidden_layers_rest[0](layers.concatenate([hl1_A, hl1_B, hl1_AB]))
        hl3 = self.hidden_layers_rest[1](hl2)
        hl4 = self.hidden_layers_rest[2](layers.concatenate([hl1_AB, hl3]))
        
        L_out = self.output_layers[0](hl4)
        
        return L_out