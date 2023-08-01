import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import model as m
import math
from torch.utils.data import DataLoader
from data.num_ds import num_ds # change to universal dataset later
from model import Prob_Type

def train(prob_type: Prob_Type,
            dataset,
            criterion,
            num_in,
            num_out,
            batch_size = 64,
            lr = .001,
            epochs = 5000):
    # print an item in the dataset
    print(dataset[0])

    # define model
    model = m.flexnet(prob_type, num_in, num_out)

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)

    # create DataLoader with your dataset and desired batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the training loop
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
        if epoch % 1000 == 0:
            print(str(epoch)+ " " + "epochs done!") # visualze results after every 1000 epochs
        print("Epoch ", epoch, "Loss ", total_loss / len(dataloader))

