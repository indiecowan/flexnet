import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

# source env/bin/activate
# deactivate

# Define the class for single layer NN
class flexnet(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, hidden_dim, num_linear_layers, output_size):
        super(flexnet, self).__init__()

        # use ordered dict to store layers for use of sequential initlization
        layers_od = OrderedDict()

        # add linear layers
        layers_od['linear_1'] = nn.Linear(input_size, hidden_dim)
        layers_od['relu_1'] = nn.ReLU()

        for i in range(num_linear_layers - 1):
            layers_od['linear_' + str(i + 2)] = nn.Linear(hidden_dim, hidden_dim)
            layers_od['relu_' + str(i + 2)] = nn.ReLU()
    
        layers_od['linear_' + str(num_linear_layers + 1)] = nn.Linear(hidden_dim, output_size)

        # add layers to sequential
        self.layers = nn.Sequential(layers_od)


    # prediction function
    def forward(self, x):
        return self.layers(x)
    
