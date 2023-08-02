import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from enum import Enum

# model type enum
class Prob_Type(Enum):
    B = 1
    C = 2
    R = 3

# source env/bin/activate
# deactivate

# Define the class for single layer NN
class flexnet(torch.nn.Module):    
    # Constructor
    def __init__(self, type: Prob_Type, input_size: int, output_size: int, num_hidden_layers: int = 1, hidden_dim: int = 10):
        super(flexnet, self).__init__()

        # use ordered dict to store layers for use of sequential initlization
        layers_od = OrderedDict()


        # add linear layers
        if num_hidden_layers == 0:
            layers_od['linear_1'] = nn.Linear(input_size, output_size)
            self.layers = nn.Sequential(layers_od)
        
        else:
            layers_od['linear_1'] = nn.Linear(input_size, hidden_dim)
            layers_od['relu_1'] = nn.ReLU()

            # hidden layers
            for i in range(num_hidden_layers - 2):
                layers_od['linear_' + str(i + 2)] = nn.Linear(hidden_dim, hidden_dim)
                layers_od['relu_' + str(i + 2)] = nn.ReLU()
        
            layers_od['linear_' + str(num_hidden_layers + 1)] = nn.Linear(hidden_dim, output_size)

        if type == Prob_Type.B:
            layers_od['sigmoid'] = nn.Sigmoid()
        elif type == Prob_Type.C:
            layers_od['softmax'] = nn.Softmax(dim=1)

        # add layers to sequential
        self.layers = nn.Sequential(layers_od)


    # prediction function
    def forward(self, x):
        return self.layers(x)
    
