import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        EXTRA_STATES = 2
        super(DQN, self).__init__()
        self.action_size = action_size

        h = 2*action_size #hidden dimension
        num_of_channels = [3, 16, 64]
        kernel_sizes = [3, 1]
        stride_sizes = [1, 1]

        self.convNN = nn.Sequential(
            nn.Conv2d(num_of_channels[0], num_of_channels[1], kernel_size=kernel_sizes[0]),
            nn.BatchNorm2d(num_of_channels[1]),
            nn.ReLU(),
            nn.Conv2d(num_of_channels[1], num_of_channels[2], kernel_size=kernel_sizes[1]),
            nn.BatchNorm2d(num_of_channels[2]),
            nn.ReLU())
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.neuralnet = nn.Sequential(
            nn.Linear(num_of_channels[-1] + EXTRA_STATES, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, self.action_size))

    def forward(self, x):
        y = self.convNN(x[0])
        y = self.avgpool(y)
        y = y.flatten(start_dim=1)
        y = torch.cat((y, x[1]), dim=1)
        y = self.neuralnet(y)
        return y