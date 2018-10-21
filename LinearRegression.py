import torch
from torch import nn
from torch.autograd import Variable
from torch import optim


# Preparing data for regression
x = Variable(torch.Tensor([[-1. ], [-0.9], [-0.8], [-0.7], [-0.6], [-0.5],
                           [-0.4], [-0.3], [-0.2], [-0.1], [ 0. ], [ 0.1],
                           [ 0.2], [ 0.3], [ 0.4], [ 0.5], [ 0.6], [ 0.7],
                           [ 0.8], [ 0.9], [ 1. ]]), requires_grad=False)
y_true = Variable(torch.Tensor([[32.6 ], [31.62], [30.64], [29.66], [28.68],
                                [27.7 ], [26.72], [25.74], [24.76], [23.78],
                                [22.8 ], [21.82], [20.84], [19.86], [18.88],
                                [17.9 ], [16.92], [15.94], [14.96], [13.98],
                            [13.  ]]), requires_grad=False)

# Creating a Linear Regression Module
# which just involves multiplying a weight
# with the feature value and adding a bias term
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        # Since there is only a single feature value,
        # and the output is a single value, we use
        # the Linear module with dimensions 1 X 1.
        # It adds a bias term by default
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


linreg = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(linreg.parameters(), lr=0.1)

for i in range(100):
    y_pred = linreg(x)
    loss = criterion(y_true, y_pred)
    print('Loss:', loss.data, 'Parameters:',
          list(map(lambda x: x.data, linreg.parameters())))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
