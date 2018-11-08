import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable


xs = np.array([[0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.], [0., 0., 1., 1.],
              [0., 1., 0., 0.], [0., 1., 0., 1.], [0., 1., 1., 0.], [0., 1., 1., 1.],
              [1., 0., 0., 0.], [1., 0., 0., 1.], [1., 0., 1., 0.], [1., 0., 1., 1.],
              [1., 1., 0., 0.], [1., 1., 0., 1.], [1., 1., 1., 0.], [1., 1., 1., 1.]], dtype=float)

ys = np.array([[0.], [1.], [1.], [0.], [1.], [0.], [0.], [1.], [1.], [0.], [0.], [1.],
               [0.], [1.], [1.], [0.]], dtype=float)

x_var = Variable(Tensor(xs), requires_grad=False)
y_var = Variable(Tensor(ys), requires_grad=False)

EPOCHS = 1000


# Helper function to train the network
def train(model, x_train, y_train, criterion, optimizer, epochs):

    for i in range(epochs):
        # Make predictions (forward propagation)
        y_pred = model(x_train)

        # Compute and print the loss every hundred epochs
        loss = criterion(y_pred, y_train)
        if i % 100 == 0:
            print('Loss:', loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Create a reusable module
# PyTorch makes writing modular OO code extremely easy
class LinearBlock(nn.Module):

    def __init__(self, in_nums, out_nums, activation):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_nums, out_nums)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))


class FullyConnectedNet(nn.Module):

    def __init__(self, input_size, neurons, activations):
        super(FullyConnectedNet, self).__init__()

        # For now, we will have a linear layer followed by an activation function
        assert len(neurons) == len(activations), 'Number of activations must be equal to the number of activations'

        # We will need a list of blocks cascaded one after the other, so we keep them in a ModuleList instead of a Python list
        self.blocks = nn.ModuleList()

        previous = input_size
        for i in range(len(neurons)):
            self.blocks.append(LinearBlock(previous, neurons[i], activations[i]))
            previous = neurons[i]

    def forward(self, x):
        "Pass the input through each block"
        for block in self.blocks:
            x = block(x)

        return x


# Crete a network with 2 hidden layers and 1 output layer, with sigmoid activations
fcnet01 = FullyConnectedNet(4, # We have a four dimensional input
                            [4, 5, 1], # We two hidden layers with 4 neurons each, and an output layer
                                       # with 1 neuron
                            [nn.ReLU(), nn.Sigmoid(), nn.Sigmoid()] # Using sigmoid for activation
                            )
print(fcnet01)
# Since it's a 0-1 problem, we will use Binary Cross Entropy as our loss function
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(fcnet01.parameters(), lr=0.01)

# Then, our usual training loop
train(fcnet01, x_var, y_var, criterion, optimizer, EPOCHS)

fcnet02 = FullyConnectedNet(4,
                            [8, 10, 10, 8, 1],
							[nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid()])