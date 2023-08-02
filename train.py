import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import model as m
import math
from torch.utils.data import DataLoader
from data.num_ds import num_ds # change to universal dataset later
from model import Prob_Type

def train(prob_type: Prob_Type,
            train_ds,
            dev_ds,
            criterion,
            num_in,
            num_out,
            batch_size = 64,
            lr = .001,
            epochs = 5000):
    # print an item in the dataset
    # print(dataset[0])

    # define model
    model = m.flexnet(prob_type, num_in, num_out)

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)

    # create DataLoader with your dataset and desired batch size
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=True)

    # Define the training loop
    for epoch in range(epochs):
        total_loss = 0
        for X, Y in train_dl:
            Yhat = model(X)
            loss = criterion(Yhat, Y)
            if math.isnan(loss.item()):
                print("Loss is NaN.")
                return
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() 
        if epoch % 500 == 0:
            print("Epoch ", epoch, "Loss ", total_loss / len(train_dl))
        if epoch % 1000 == 0:
            print(str(epoch)+ " " + "epochs done!") # visualze results after every 1000 epochs
            # calculate dev loss
            model.eval()
            dev_loss = 0
            with torch.no_grad():
                for X, Y in dev_dl:
                    Yhat = model(X)
                    loss = criterion(Yhat, Y)
                    dev_loss += loss.item()
            print("Dev loss: ", dev_loss / len(dev_dl))
            model.train()
    print("Epoch ", epoch, "Loss ", total_loss / len(train_dl))

    # calculate dev loss
    model.eval()
    dev_loss = 0
    with torch.no_grad():
        for X, Y in dev_dl:
            Yhat = model(X)
            loss = criterion(Yhat, Y)
            dev_loss += loss.item()
    print("Dev loss: ", dev_loss / len(dev_dl))


