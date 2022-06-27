import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(10 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input, target):
        x = self.conv1(input)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)


        x = x.view(-1, 10 * 4 * 4)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        loss = self.loss_function(x, target)
        return x, loss