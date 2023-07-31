import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

# source env/bin/activate
# deactivate

# Define the class for single layer NN
class flexible_net(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, hidden_dim, num_linear_layers, output_size):
        super(flexible_net, self).__init__()

        # use ordered dict to store layers for use of sequential initlization
        layers_od = OrderedDict()

        # add linear layers
        layers_od['linear_1'] = nn.Linear(input_size, hidden_dim)

        for i in range(num_linear_layers - 1):
            layers_od['linear_' + str(i + 2)] = nn.Linear(hidden_dim, hidden_dim)
    
        layers_od['linear_' + str(num_linear_layers + 1)] = nn.Linear(hidden_dim, output_size)

        # add layers to sequential
        self.layers = nn.Sequential(layers_od)


    # prediction function
    def forward(self, x):
        return self.layers(x)
    


# generate synthetic the data
X = torch.arange(-30, 30, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] <= -10)] = 1.0
Y[(X[:, 0] > -10) & (X[:, 0] < 10)] = 0.5
Y[(X[:, 0] > 10)] = 0
print(X)
print(Y)

# run model with data
model = flexible_net(1, 10, 3, 1)
Yhat = model(X)
Y = Y.view(-1, 1)

# find loss
criterion = nn.MSELoss()
loss = criterion(Yhat, Y)
print("loss ", + loss.item())