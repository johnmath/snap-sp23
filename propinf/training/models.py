import torch

from torch import nn
from typing import List
import torch.nn.functional as F


class NeuralNet(nn.Module):
    """PyTorch implementation of a multilayer perceptron with ReLU activations"""

    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int] = [64],
        num_classes: int = 2,
        dropout: bool = False,
    ):
        super(NeuralNet, self).__init__()
        self._input_dim = input_dim
        self._layer_sizes = layer_sizes
        self._num_classes = num_classes

        layers = [nn.Linear(input_dim, layer_sizes[0])]
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout())

        # Initialize all layers according to sizes in list
        for i in range(len(self._layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout())
        layers.append(nn.Linear(layer_sizes[-1], num_classes))

        # Wrap layers in ModuleList so PyTorch
        # can compute gradients
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LogisticRegression(nn.Module):
    """PyTorch implementation of logistic regression
    Note: If you are using nn.CrossEntropyLoss(), set using_ce_loss=True.
    Pytorch's implementation of CrossEntropyLoss computes the softmax
    of the output before categorical cross entropy. Therefore, using the
    sigmoid activation function in the forward call is unneccessary"""

    def __init__(
        self, input_dim: int, num_classes: int = 2, using_ce_loss: bool = True
    ):
        super(LogisticRegression, self).__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes
        self._linear = torch.nn.Linear(self._input_dim, self._num_classes)
        self._using_ce_loss = using_ce_loss

    def forward(self, x):
        if self._using_ce_loss:
            return self._linear(x)
        else:
            return nn.functional.softmax(self._linear(x), dim=1)


class SimpleCNN(nn.Module):
    def __init__(self):
        """
        Simple CNN model for CIFAR-10.
        3 Convolutions, 2 Fully Connected Layers
        Use dropout to prevent overfitting
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
