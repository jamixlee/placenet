import torch
from torch import nn
import math

class Tower(nn.Module):
    def __init__(self, x_ch, v_ch, r_ch):
        """
        Network that generates a condensed representation vector from a joint input of image and viewpoint.

        Parameters:
        x_ch: number of color channels in input image
        v_ch: dimensions of the viewpoint vector
        r_ch: dimensions of representation
        pool: whether to pool representation
        """
        super(Tower, self).__init__()
        self.r_size = int(math.sqrt(r_ch))  # common value in this experiment

        self.conv1 = nn.Conv2d(x_ch,      r_ch,    kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(r_ch,      r_ch,    kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(r_ch,      r_ch//2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(r_ch//2,   r_ch,    kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(r_ch+v_ch, r_ch,    kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(r_ch+v_ch, r_ch//2, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(r_ch//2,   r_ch,    kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(r_ch,      r_ch,    kernel_size=1, stride=1)

        self.avgpool = nn.AvgPool2d(self.r_size)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=x_ch, eps=1e-03)
        self.dropout = torch.nn.Dropout2d(p=0.2)  # p: dropout_probability


    def forward(self, x, v):
        """
        Send an (image, viewpoint) pair into the network to generate a representation

        Parameters:
        x: image
        v: viewpoint (x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch))
        r: representation
        """
        # Increase dimensions
        v = v.view(v.size(0), -1, 1, 1)
        v = v.repeat(1, 1, self.r_size, self.r_size)

        # First skip-connected convolution block
        skip_in = self.relu(self.conv1(x))
        skip_out = self.relu(self.conv2(skip_in))

        r = self.relu(self.conv3(skip_in))
        r = self.relu(self.conv4(r)) + skip_out

        # Second skip-connected convolution block (merged)
        skip_in = torch.cat([r, v], dim=1)
        skip_out = self.relu(self.conv5(skip_in))

        r = self.relu(self.conv6(skip_in))
        r = self.relu(self.conv7(r)) + skip_out

        r = self.relu(self.conv8(r))

        return r
