import torch
from torch import nn
import numpy as np
import h5py

hf = h5py.File('data.h5', 'r')
X_train = torch.Tensor(np.array(hf.get("X_train")).T)
y_train = torch.Tensor(np.array(hf.get("y_train")).reshape(-1, 1))
X_test = torch.Tensor(np.array(hf.get("X_test")).T)
y_test = torch.Tensor(np.array(hf.get("y_test")).reshape(-1, 1))

k = X_train.shape[1]

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(k, 1)
criterion = nn.L1Loss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

# Testing: what if we know the true weights?
# beta = torch.rand(k)
# y_train = (X_train @ beta).reshape(-1, 1)

for epoch in range(2000):
    optimiser.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimiser.step()
    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch,loss.item()))
