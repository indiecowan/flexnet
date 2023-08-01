import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import model as m

# generate synthetic the data
X = torch.arange(-30, 30, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] <= -10)] = 1.0
Y[(X[:, 0] > -10) & (X[:, 0] < 10)] = 0.5
Y[(X[:, 0] > 10)] = 0
print(X)
print(Y)

# run model with data
model = m.flexnet(1, 10, 3, 1)
Yhat = model(X)
Y = Y.view(-1, 1)

# find loss
criterion = nn.MSELoss()
loss = criterion(Yhat, Y)
print("loss ", + loss.item())

# train
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# Define the training loop
epochs=5000
cost = []
total=0
for epoch in range(epochs):
    total=0
    epoch = epoch + 1
    for x, y in zip(X, Y):
        yhat = model(x.unsqueeze(0))
        loss = criterion(yhat, y.unsqueeze(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # get total loss 
        total+=loss.item() 
    if epoch % 500 == 0:
        print("Epoch ", epoch, "Loss ", total / X.shape[0])
    cost.append(total)
    if epoch % 1000 == 0:
        print(str(epoch)+ " " + "epochs done!") # visualze results after every 1000 epochs   
        # plot the result of function approximator
        plt.plot(X.numpy(), model(X).detach().numpy())
        plt.plot(X.numpy(), Y.numpy(), 'm')
        plt.xlabel('x')
        plt.show()