import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import model as m
import math
from torch.utils.data import DataLoader
import data.num_ds as ds # change to universal dataset later

# make dataset object
dataset = ds.num_ds('data/pd_1/patient_data.csv')
# print an item in the dataset
print(dataset[0])

# create DataLoader with your dataset and desired batch size
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = m.flexnet(22, 10, 3, 3)  # assume 22 inputs, 10 hidden units, 3 linear layers, 3 output classes

# Define the loss
criterion = nn.CrossEntropyLoss()  # use CrossEntropyLoss for multi-class classification

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Define the training loop
epochs=5000
cost = []

for epoch in range(epochs):
    total_loss = 0
    for X, Y in dataloader:
        Yhat = model(X)
        loss = criterion(Yhat, Y)
        if math.isnan(loss.item()):
            print("Loss is NaN.")
            break
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item() 
    if epoch % 500 == 0:
        print("Epoch ", epoch, "Loss ", total_loss / len(dataloader))
    cost.append(total_loss)
    if epoch % 1000 == 0:
        print(str(epoch)+ " " + "epochs done!") # visualze results after every 1000 epochs   
        # plot the result of function approximator
        plt.plot(X.numpy(), model(X).detach().numpy())
        plt.plot(X.numpy(), Y.numpy(), 'm')
        plt.xlabel('x')
        plt.show()
