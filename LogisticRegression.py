import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# For plotting
import matplotlib.pyplot as plt
import seaborn as sn
from pylab import rcParams

NUM_EPOCHS = 2000

sn.set_style('darkgrid')    # Set the theme of the plot
rcParams['figure.figsize'] = 18, 10  # Set the size of the plot image

# Creating a Logisitic Regression Module
# which just involves multiplying a weight
# with the feature value and adding a bias term
class LogisticRegression(nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        # Since there is only a single feature value,
        # and the output is a single value, we use
        # the Linear module with dimensions 1 X 1.
        # It adds a bias term by default
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return F.sigmoid(self.linear(x))


# Preparing data for classification
xs = np.linspace(-1, 1, 21)
ys = np.array([0.0 if x < 0 else 1 for x in xs])

# Plot the points to see if everything is right
plt.plot(xs, ys, 'ro')
plt.show()

# We use torch's `from_numpy` function to convert
# our numpy arrays to Tensors. They are later wrapped
# in torch Variables
x = Variable(torch.from_numpy(xs), requires_grad=False)
y_true = Variable(torch.from_numpy(ys), requires_grad=False)

x = x.float()
y_true = y_true.float()

# Convert to feature matrices
x = x.reshape(-1, 1)
y_true = y_true.reshape(-1, 1)

logreg = LogisticRegression()
criterion = nn.BCELoss()   # Using the Binary Cross Entropy loss
                           # since it's a classfication task
optimizer = optim.SGD(logreg.parameters(), lr=0.1)

for i in range(NUM_EPOCHS):
    y_pred = logreg(x)
    loss = criterion(y_pred, y_true)
    print('Loss:', loss.data, 'Parameters:',
          list(map(lambda x: x.data, logreg.parameters())))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

params = list(map(lambda x: x.data, logreg.parameters()))
m = params[0].numpy()[0][0]
c = params[1].numpy()[0]
print(m, c)
y = m * x.numpy() + c
plt.plot(xs[np.where(xs >= 0)], ys[ys == 1.0], 'bo', label='1')
plt.plot(xs[np.where(xs < 0)], ys[ys == 0.0], 'ro', label='0')
plt.plot(x.numpy(), y, 'g', label='Decision Boundary')
plt.show()