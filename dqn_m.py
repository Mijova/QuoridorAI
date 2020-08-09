import torch
import torch.nn as nn
import torch.nn.functional as F

# DQN model : 2 conv layers and 2 fc layers
# Send positions of players and walls to conv layers
# Add number of remaining walls for both players to fc layers
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.action_size = action_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.ada = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64+2, 64)
        self.fc2 = nn.Linear(64, self.action_size)

    def forward(self, x):
        y = x[:,3:,0,0].view(-1) # extract the number of remaining walls for both players
        x = x[:,0:3,:,:] # extract positions of players and walls
        
        x = self.conv1(x)
        x = F.relu(self.norm1(x))
        x = self.conv2(x)
        x = F.relu(self.norm2(x))
        x = self.ada(x)
        x = x.view(-1)

        x = torch.cat((x,y),0)
        x = self.fc1(x)
        x = self.fc2(x)
        return x