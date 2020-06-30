import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, rows, columns, in_channels=1):
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=16 * 6 * 7, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=columns)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)

        t = self.conv2(t)
        t = F.relu(t)

        t = torch.flatten(t, start_dim=1)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.out(t)

        return t
