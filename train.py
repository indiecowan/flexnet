import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import model as m
import math
from torch.utils.data import DataLoader
from data.num_ds import num_ds # change to universal dataset later
from model import Prob_Type
import wandb

def train(prob_type: Prob_Type,
            train_ds,
            dev_ds,
            criterion,
            num_in,
            num_out,
            num_hid_layers = 2,
            hid_dim = 10,
            batch_size = 64,
            lr = .001,
            epochs = 5000):
    # print an item in the dataset
    # print(dataset[0])

    # define model
    model = m.flexnet(prob_type, num_in, num_out, num_hid_layers, hid_dim)

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)

    # define scheduler reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=.1)

    # create DataLoader with your dataset and desired batch size
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=True)

    # Define the training loop
    for epoch in range(epochs):
        total_loss = 0
        # train over whole ds
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
        
        # log train loss after training is done
        wandb.log({"train_loss": total_loss / len(train_dl), "lr": optimizer.param_groups[0]["lr"]})

        # print train loss every 100 epochs
        if epoch % 100 == 0:
            print("epoch ", epoch, ":\n    train loss ", total_loss / len(train_dl))

        # calculate dev loss
        # print(str(epoch)+ " " + "epochs done!") # visualze results after every 1000 epochs
        # calculate dev loss
        # switch to eval mode
        model.eval()
        dev_loss = 0
        with torch.no_grad():
            for X, Y in dev_dl:
                Yhat = model(X)
                loss = criterion(Yhat, Y)
                dev_loss += loss.item()
        
        # step scheduler
        scheduler.step(dev_loss / len(dev_dl))

        # print dev loss every 100 epochs
        if epoch % 100 == 0:
            print("   dev loss: ", dev_loss / len(dev_dl))

        # log dev loss
        wandb.log({"dev_loss": dev_loss / len(dev_dl)})
        # switch back to train mode
        model.train()

    # print last results
    print("epoch ", epoch, ":\n    train loss ", total_loss / len(train_dl))

    # calculate dev loss
    model.eval()
    dev_loss = 0
    with torch.no_grad():
        for X, Y in dev_dl:
            Yhat = model(X)
            loss = criterion(Yhat, Y)
            dev_loss += loss.item()

    # pritn final dev loss
    print("    dev loss: ", dev_loss / len(dev_dl))


def sweep_train(prob_type: Prob_Type,
                train_ds,
                dev_ds,
                criterion,
                num_in,
                num_out):

    # Initialize a new wandb run
    with wandb.init() as run:
        # Then call your train function with run.config values:
        train(prob_type, train_ds, dev_ds, criterion, num_in, num_out, num_hid_layers=run.config.hid_layers, hid_dim=run.config.hid_dim, batch_size=run.config.batch_size, lr=run.config.lr, epochs=run.config.epochs)



